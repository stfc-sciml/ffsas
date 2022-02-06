#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# system.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" G-based system for SAS modelling and inversion """

import sys

import h5py
import torch
from scipy import optimize

from ffsas import torch_dtype
from ffsas.utils import MultiLevelLogger, _form_batch_ids


def diag_indices(n):
    """ torch does not support this """
    return (range(n),) * 2


def _to_tensor(g, device='cpu'):
    """ convert g to torch.Tensor """
    if isinstance(g, torch.Tensor):
        return g.to(dtype=torch_dtype, device=device)
    else:
        return torch.tensor(g, dtype=torch_dtype, device=device)


def _order_of_magnitude(v):
    """ get order of magnitude of v """
    return 10 ** torch.floor(torch.log10(torch.abs(v))).item()


class SASGreensSystem:
    def _get_x0(self, xi0, b0, w_dict_init=None):
        """ get initial guess x0 """
        if w_dict_init is None:
            w_dict_init = {}
        x0 = torch.empty(sum(self._s_dims) + 2, dtype=torch_dtype)
        pos = 0
        for i, s_dim in enumerate(self._s_dims):
            if self._par_keys[i] not in w_dict_init.keys():
                # using uniform distributions
                x0[pos:pos + s_dim] = torch.sqrt(
                    torch.ones(s_dim, dtype=torch_dtype) / s_dim)
            else:
                w_init = w_dict_init[self._par_keys[i]]
                x0[pos:pos + s_dim] = torch.sqrt(w_init / w_init.sum())
            pos += s_dim
        x0[-2] = xi0
        x0[-1] = b0
        return x0.numpy()

    def _extract(self, x, device):
        """ extract (s, xi, b) from x """
        s_list = []
        pos = 0
        for s_dim in self._s_dims:
            s = _to_tensor(x[pos:pos + s_dim], device=device)
            pos += s_dim
            s_list.append(s)
        xi = x[-2]
        b = x[-1]
        return s_list, xi, b

    def _g_dot_w(self, g, w_list, skips):
        """ perform successive G.w1...wk...wn """
        # non-skip indices
        non_skips = list(range(len(w_list)))
        for i in skips:
            non_skips.remove(i)

        # all skipped
        if len(non_skips) == 0:
            return g

        # outer product of w
        w_outer = w_list[non_skips[0]]
        for i in non_skips[1:]:
            w_outer = torch.einsum('...i,j->...ij', w_outer, w_list[i])

        # contraction
        gid = [i + self._nq for i in non_skips]
        wid = list(range(len(non_skips)))
        return torch.tensordot(g, w_outer, dims=(gid, wid))

    def _g_dot_s2(self, g, s_list, skips):
        """ perform successive G.s1^2...sk^2...sn^2 """
        w_list = [None if i in skips else s ** 2 for i, s in enumerate(s_list)]
        return self._g_dot_w(g, w_list, skips)

    def _obj_func(self, x):
        """ objective """
        # extract
        s_list, xi, b = self._extract(x, device=self._device)
        xi_m, b_m = xi * self._xi_mag, b * self._b_mag

        # initialize Gs2
        Gs2 = torch.empty(self._q_dims, dtype=torch_dtype,
                          device=self._device)
        # compute Gs2 by batch
        for G_id in self._batch_ids:
            # batch
            g = _to_tensor(self._G[G_id], device=self._device)
            # Gs2
            Gs2[G_id] = self._g_dot_s2(g, s_list, skips=[])

        # objective
        eps = (xi_m * Gs2 + b_m - self._mu) / self._nv
        L = .5 * torch.tensordot(eps, eps, dims=self._nq)
        return L.item()

    def _form_jac_hess(self, x):
        """ form Jacobian and Hessian"""
        # extract
        s_list, xi, b = self._extract(x, device=self._device)
        xi_m, b_m = xi * self._xi_mag, b * self._b_mag

        # initialize Gs2_ij
        Gs2_ij = []
        for i in range(0, self._ns):
            shape = self._q_dims + [self._s_dims[i], -1]
            Gs2_ij_i = [torch.empty(0)] * self._ns
            for j in range(i + 1, self._ns):
                shape[-1] = self._s_dims[j]
                Gs2_ij_i[j] = torch.empty(
                    shape, dtype=torch_dtype, device=self._device)
            Gs2_ij.append(Gs2_ij_i)

        # compute Gs2_ij by batch
        for G_id in self._batch_ids:
            # copy s_list
            s_list_remain = s_list.copy()
            # batch
            g = _to_tensor(self._G[G_id], device=self._device)
            # Gs2_ij
            for i in range(0, self._ns - 1):
                s_list_remain_j = s_list_remain.copy()
                g_j = g.clone()
                for j in range(self._ns - 1, i, -1):
                    Gs2_ij[i][j][G_id] = self._g_dot_s2(
                        g_j, s_list_remain_j,
                        skips=[0, len(s_list_remain_j) - 1])
                    if j != i + 1:
                        g_j = torch.tensordot(g_j, s_list_remain_j.pop() ** 2,
                                              dims=1)
                if i != self._ns - 2:
                    g = torch.tensordot(g, s_list_remain.pop(0) ** 2,
                                        dims=([self._nq], [0]))

        # Gs2_i based on Gs2_ij
        Gs2_i = [torch.empty(0)] * self._ns
        # all except last one
        for i in range(0, self._ns - 1):
            Gs2_i[i] = torch.tensordot(Gs2_ij[i][i + 1],
                                       s_list[i + 1] ** 2, dims=1)
        # last one
        if self._ns > 1:
            Gs2_i[-1] = torch.tensordot(Gs2_ij[-2][-1],
                                        s_list[-2] ** 2, dims=([-2], [0]))
        else:
            # no Gs2_ij calculated when ns = 1
            Gs2_i[0] = _to_tensor(self._G[:], device=self._device)
        # Gs2 based on Gs2_i
        Gs2 = torch.tensordot(Gs2_i[0], s_list[0] ** 2, dims=1)

        # eps
        eps = (xi_m * Gs2 + b_m - self._mu) / self._nv
        deps_dxi = self._xi_mag * Gs2 / self._nv
        deps_db = self._b_mag / self._nv

        # Jacobian
        nv_indexed = self._nv[self._q_slices + (None,)]
        J = torch.empty(len(x), dtype=torch_dtype, device=self._device)
        pos = 0
        for i, s_dim in enumerate(self._s_dims):
            deps_ds = 2. * xi_m * Gs2_i[i] * s_list[i] / nv_indexed
            dL_ds = torch.tensordot(eps, deps_ds, dims=self._nq)
            J[pos:pos + s_dim] = dL_ds
            pos += s_dim
        J[-2] = torch.tensordot(eps, deps_dxi, dims=self._nq).item()
        J[-1] = torch.tensordot(eps, deps_db, dims=self._nq).item()
        self._jac_save = J.to('cpu').numpy()

        # Hessian
        q_id = (list(range(self._nq)), list(range(self._nq)))
        H = torch.zeros((len(x), len(x)), dtype=torch_dtype,
                        device=self._device)
        pos_i = 0
        for i in range(0, self._ns):
            ##################
            # Matrix terms:  #
            # 1) d^2L/ds/ds  #
            ##################
            slice_i = slice(pos_i, pos_i + self._s_dims[i])
            deps_ds_i = 2. * xi_m * Gs2_i[i] * s_list[i] / nv_indexed
            # diagonal part of ε x d^2ε/ds^2
            H[slice_i, slice_i][diag_indices(self._s_dims[i])] += \
                torch.tensordot(eps, deps_ds_i / s_list[i], dims=self._nq)
            # diagonal part of dε/ds x dε/ds
            H[slice_i, slice_i] += torch.tensordot(
                deps_ds_i, deps_ds_i, dims=q_id)
            pos_j = sum(self._s_dims[0:i + 1])
            for j in range(i + 1, self._ns):
                slice_j = slice(pos_j, pos_j + self._s_dims[j])
                deps_ds_j = 2. * xi_m * Gs2_i[j] * s_list[j] / nv_indexed
                # off-diagonal part of dε/ds x dε/ds
                H[slice_i, slice_j] += torch.tensordot(
                    deps_ds_i, deps_ds_j, dims=q_id)
                # off-diagonal term of ε x d^2ε/ds^2
                Gs2_ij_sij = Gs2_ij[i][j] * s_list[j]
                Gs2_ij_sij = torch.transpose(Gs2_ij_sij, -1, -2) * s_list[i]
                Gs2_ij_sij = torch.transpose(Gs2_ij_sij, -1, -2)
                # self._nv goes from Gs2_ij_sij to eps
                H[slice_i, slice_j] += (4. * xi_m * torch.tensordot(
                    eps / self._nv, Gs2_ij_sij, dims=self._nq))
                pos_j += self._s_dims[j]
            pos_i += self._s_dims[i]

            ##################
            # Vector terms:  #
            # 1) d^2L/ds/dxi #
            # 2) d^2L/ds/db  #
            ##################
            # ε x d^2ε/ds/dxi
            H[slice_i, -2] += torch.tensordot(eps, deps_ds_i / xi,
                                              dims=self._nq)
            # dε/dxi x dε/ds
            H[slice_i, -2] += torch.tensordot(
                deps_dxi, deps_ds_i, dims=self._nq)
            # dε/db x dε/ds
            H[slice_i, -1] += torch.tensordot(
                deps_db, deps_ds_i, dims=self._nq)

        ###################
        # Scalar terms:   #
        # 1) d^2L/dxi/dxi #
        # 2) d^2L/dxi/db  #
        # 3) d^2L/db/db   #
        ###################
        # dε/dxi x dε/dxi
        H[-2, -2] += torch.tensordot(deps_dxi, deps_dxi, dims=self._nq)
        # dε/dxi x dε/db
        H[-2, -1] += torch.tensordot(deps_dxi, deps_db, dims=self._nq)
        # dε/db x dε/db
        H[-1, -1] += torch.tensordot(deps_db, deps_db, dims=self._nq)

        # copy upper to lower
        lower_ids = torch.tril_indices(len(x), len(x), offset=-1)
        H[tuple(lower_ids)] = H.t()[tuple(lower_ids)]  # tuple must be used!
        self._hess_save = H.to('cpu').numpy()

        # save X to check
        self._x_save = x

    def _jac_func(self, x):
        """ Jacobian """
        # check call order
        assert self._jac_turn
        self._jac_turn = False
        # form jac and hess
        self._form_jac_hess(x)
        return self._jac_save

    def _hess_func(self, x):
        """ Hessian """
        # check call order
        assert not self._jac_turn
        self._jac_turn = True
        # check x
        assert (x == self._x_save).all()
        return self._hess_save

    def _constr_s2(self, x):
        """ constraint s^2-1=0"""
        s_list, _, _ = self._extract(x, device='cpu')
        err = torch.empty(self._ns, dtype=torch_dtype)
        for i, s in enumerate(s_list):
            err[i] = torch.dot(s, s) - 1.
        return err.numpy()

    def _constr_s2_J(self, x):
        """ Jacobian of s^2-1=0"""
        s_list, _, _ = self._extract(x, device='cpu')
        J = torch.zeros((self._ns, len(x)), dtype=torch_dtype)
        pos = 0
        for i, s in enumerate(s_list):
            J[i, pos:pos + len(s)] = 2. * s
            pos += len(s)
        return J.numpy()

    def _constr_s2_H(self, x, v):
        """ Hessian of (s^2-1).v """
        H = torch.zeros((len(x), len(x)), dtype=torch_dtype)
        pos = 0
        for i, s_dim in enumerate(self._s_dims):
            H[pos:pos + s_dim, pos:pos + s_dim][diag_indices(s_dim)] = 2. * v[i]
            pos += s_dim
        return H.numpy()

    def _intensity(self, w_list, xi, b):
        """ compute intensity """
        # initialize Gw
        Gw = torch.empty(self._q_dims, dtype=torch_dtype,
                         device=self._device)
        # compute Gw by batch
        for G_id in self._batch_ids:
            # batch
            g = _to_tensor(self._G[G_id], device=self._device)
            # Gw
            Gw[G_id] = self._g_dot_w(g, w_list, skips=[])

        # intensity
        return xi * Gw + b

    def _intensity_sensitivity(self, w_list, xi, b):
        """ compute intensity and sensitivity """
        ############
        # Gw terms #
        ############
        # initialize Gw_ij
        Gw_ij = []
        for i in range(0, self._ns):
            shape = self._q_dims + [self._s_dims[i], -1]
            Gw_ij_i = [torch.empty(0)] * self._ns
            for j in range(i + 1, self._ns):
                shape[-1] = self._s_dims[j]
                Gw_ij_i[j] = torch.empty(
                    shape, dtype=torch_dtype, device=self._device)
            Gw_ij.append(Gw_ij_i)

        # compute Gw_ij by batch
        for G_id in self._batch_ids:
            # copy w_list
            w_list_remain = w_list.copy()
            # batch
            g = _to_tensor(self._G[G_id], device=self._device)
            # Gw_ij
            for i in range(0, self._ns - 1):
                w_list_remain_j = w_list_remain.copy()
                g_j = g.clone()
                for j in range(self._ns - 1, i, -1):
                    Gw_ij[i][j][G_id] = self._g_dot_w(
                        g_j, w_list_remain_j,
                        skips=[0, len(w_list_remain_j) - 1])
                    if j != i + 1:
                        g_j = torch.tensordot(g_j, w_list_remain_j.pop(),
                                              dims=1)
                if i != self._ns - 2:
                    g = torch.tensordot(g, w_list_remain.pop(0),
                                        dims=([self._nq], [0]))

        # Gw_i based on Gw_ij
        Gw_i = [torch.empty(0)] * self._ns
        # all except last one
        for i in range(0, self._ns - 1):
            Gw_i[i] = torch.tensordot(Gw_ij[i][i + 1],
                                      w_list[i + 1], dims=1)
        # last one
        if self._ns > 1:
            Gw_i[-1] = torch.tensordot(Gw_ij[-2][-1],
                                       w_list[-2], dims=([-2], [0]))
        else:
            # no Gw_ij calculated when ns = 1
            Gw_i[0] = _to_tensor(self._G[:], device=self._device)
        # Gw based on Gw_i
        Gw = torch.tensordot(Gw_i[0], w_list[0], dims=1)

        ##############
        # I, J, H, X #
        ##############

        # intensity
        its = xi * Gw + b

        # eps
        eps = (its - self._mu) / self._nv
        deps_dxi = Gw / self._nv
        deps_db = 1. / self._nv

        # Jacobian
        len_x = sum(self._s_dims) + 2
        nv_indexed = self._nv[self._q_slices + (None,)]
        J = torch.empty(len_x, dtype=torch_dtype, device=self._device)
        pos = 0
        for i, s_dim in enumerate(self._s_dims):
            deps_dw = xi * Gw_i[i] / nv_indexed
            dL_dw = torch.tensordot(eps, deps_dw, dims=self._nq)
            J[pos:pos + s_dim] = dL_dw
            pos += s_dim
        J[-2] = torch.tensordot(eps, deps_dxi, dims=self._nq).item()
        J[-1] = torch.tensordot(eps, deps_db, dims=self._nq).item()

        # Hessian
        q_id = (list(range(self._nq)), list(range(self._nq)))
        H = torch.zeros((len_x, len_x), dtype=torch_dtype, device=self._device)
        pos_i = 0
        for i in range(0, self._ns):
            ##################
            # Matrix terms:  #
            # 1) d^2L/dw/dw  #
            ##################
            slice_i = slice(pos_i, pos_i + self._s_dims[i])
            deps_dw_i = xi * Gw_i[i] / nv_indexed
            # diagonal part of dε/dw x dε/dw
            H[slice_i, slice_i] += torch.tensordot(
                deps_dw_i, deps_dw_i, dims=q_id)
            pos_j = sum(self._s_dims[0:i + 1])
            for j in range(i + 1, self._ns):
                slice_j = slice(pos_j, pos_j + self._s_dims[j])
                deps_dw_j = xi * Gw_i[j] / nv_indexed
                # off-diagonal part of dε/dw x dε/dw
                H[slice_i, slice_j] += torch.tensordot(
                    deps_dw_i, deps_dw_j, dims=q_id)
                # self._nv goes from Gw_ij_sij to eps
                H[slice_i, slice_j] += (xi * torch.tensordot(
                    eps / self._nv, Gw_ij[i][j], dims=self._nq))
                pos_j += self._s_dims[j]
            pos_i += self._s_dims[i]

            ##################
            # Vector terms:  #
            # 1) d^2L/dw/dxi #
            # 2) d^2L/dw/db  #
            ##################
            # ε x d^2ε/dw/dxi
            H[slice_i, -2] += torch.tensordot(eps, deps_dw_i / xi,
                                              dims=self._nq)
            # dε/dxi x dε/dw
            H[slice_i, -2] += torch.tensordot(
                deps_dxi, deps_dw_i, dims=self._nq)
            # dε/db x dε/dw
            H[slice_i, -1] += torch.tensordot(
                deps_db, deps_dw_i, dims=self._nq)

        ###################
        # Scalar terms:   #
        # 1) d^2L/dxi/dxi #
        # 2) d^2L/dxi/db  #
        # 3) d^2L/db/db   #
        ###################
        # dε/dxi x dε/dxi
        H[-2, -2] += torch.tensordot(deps_dxi, deps_dxi, dims=self._nq)
        # dε/dxi x dε/db
        H[-2, -1] += torch.tensordot(deps_dxi, deps_db, dims=self._nq)
        # dε/db x dε/db
        H[-1, -1] += torch.tensordot(deps_db, deps_db, dims=self._nq)

        # copy upper to lower
        lower_ids = torch.tril_indices(len_x, len_x, offset=-1)
        H[tuple(lower_ids)] = H.t()[tuple(lower_ids)]  # tuple must be used!

        # form x
        x = torch.empty(len_x, dtype=torch_dtype, device=self._device)
        pos = 0
        for i, s_dim in enumerate(self._s_dims):
            x[pos:pos + s_dim] = w_list[i]
            pos += s_dim
        x[-2] = xi
        x[-1] = b

        ###############
        # sensitivity #
        ###############
        sens = torch.tensordot(H, x / J, dims=1)
        sens_w_list, sens_xi, sens_b = self._extract(sens, device=self._device)
        return its, sens_w_list, sens_xi, sens_b

    def _unpack_result(self, opt_res):
        """ unpack result """
        # extract to cpu
        s_list, xi, b = self._extract(opt_res['x'], device=self._device)

        # w list
        w_list = [s ** 2 for s in s_list]
        w_dict = {self._par_keys[i]: w.to('cpu')
                  for i, w in enumerate(w_list)}

        # unscaled or physical (xi, b)
        unscaled_xi = xi * self._xi_mag
        unscaled_b = b * self._b_mag

        if self._return_intensity_sensitivity:
            # compute intensity and sensitivity
            its, sens_w_list, sens_xi, sens_b = \
                self._intensity_sensitivity(w_list, unscaled_xi, unscaled_b)
            sens_w_dict = {self._par_keys[i]: sens_w.to('cpu')
                           for i, sens_w in enumerate(sens_w_list)}
            return {'w_dict': w_dict,
                    'xi': unscaled_xi,
                    'b': unscaled_b,
                    'sens_w_dict': sens_w_dict,
                    'sens_xi': sens_xi.item(),
                    'sens_b': sens_b.item(),
                    'I': its.to('cpu'),
                    'wct': opt_res['execution_time'],
                    'opt_res': opt_res}
        else:
            return {'w_dict': w_dict,
                    'xi': unscaled_xi,
                    'b': unscaled_b,
                    'wct': opt_res['execution_time'],
                    'opt_res': opt_res}

    def _call_back_save(self, _, opt_res):
        """ callback to save results """
        if self._save_iter is None:
            return False
        if opt_res['nit'] % self._save_iter == 0:
            res_dict = self._unpack_result(opt_res)
            res_dict['nit'] = opt_res['nit']
            self._saved_results.append(res_dict)
        return False

    def _test_jac_hess(self, xi0, b0, delta=1e-8, tol=1e-4):
        """ test Jacobian and Hessian """
        with self._logger.subproc('Testing Jacobian and Hessian'):
            # L, J, H by functions
            x = self._get_x0(xi0, b0)
            L = self._obj_func(x)
            J = self._jac_func(x)
            H = self._hess_func(x)

            # L, J, H by finite difference
            J_fd = J.copy()
            H_fd = H.copy()
            for i in range(len(x)):
                x[i] += delta
                L1 = self._obj_func(x)
                J1 = self._jac_func(x)
                x[i] -= delta
                J_fd[i] = (L1 - L) / delta
                H_fd[i] = (J1 - J) / delta

            # print J error as 0/1 map
            dJ = abs(J - J_fd)
            mJ = (abs(J) + abs(J_fd)) / 2
            self._logger.message('J map (1=error):')
            for i in range(len(x)):
                print('1' if dJ[i] > mJ[i] * tol else '0', end='')
            print()

            # print H error as 0/1 map
            dH = abs(H - H_fd)
            mH = (abs(H) + abs(H_fd)) / 2
            self._logger.message('H map (1=error, T=error):')
            for i in range(len(x)):
                for j in range(len(x)):
                    if i == j:
                        print('T' if dH[i, j] > mH[i, j] * tol else 'F', end='')
                    else:
                        print('1' if dH[i, j] > mH[i, j] * tol else '0', end='')
                print()

    ##################
    # public methods #
    ##################

    def __init__(self, G, par_keys, batch_size=None, device='cpu',
                 log_file=None, log_screen=True):
        """
        Constructing the G-based system for SAS modelling and inversion

        :param G: the Green's tensor; can be a `torch.Tensor` or
            the filename of a HDF5 file containing `G` in dataset 'G-TENSOR'
        :param par_keys: parameter keys in the order as they appear in `G`
        :param batch_size: batch size along each `q`-dimension for
            mini-batch computation of G-based terms (default=None, no
            mini-batch computation)
        :param device: computing device (default='cpu')
        :param log_file: save log to this file (default=None)
        :param log_screen: print log to stdout (default=True)
        """
        # logger
        self._logger = MultiLevelLogger()
        self._logger.activate(file_path=log_file, screen=log_screen)

        # initialization
        with self._logger.subproc('Initializing SASGreensSystem object'):
            with self._logger.subproc('Determining dimensions'):
                # shape of G
                if isinstance(G, str):
                    with h5py.File(f'{G}', 'r') as f:
                        G_shape = list(f['G-TENSOR'].shape)
                else:
                    G_shape = list(G.shape)
                # split to q and s
                self._par_keys = par_keys
                self._ns = len(par_keys)
                self._nq = len(G_shape) - self._ns
                self._q_dims = G_shape[:self._nq]
                self._s_dims = G_shape[self._nq:]
                self._q_slices = [slice(0, q_dim) for q_dim in self._q_dims]
                self._q_slices = tuple(self._q_slices)
                self._logger.message(f'q dimensions = {self._q_dims}')
                self._logger.message(f'w dimensions = {self._s_dims}')
                self._logger.message(f'parameter keys = {self._par_keys}')
                self._logger.message(f'torch dtype = {torch_dtype}')

            with self._logger.subproc('Resolving space for G'):
                if isinstance(G, str):
                    # copying file is faster than copying variable
                    G_file = h5py.File(G, 'r')
                    assert 'G-TENSOR' in G_file.keys()
                    self._G = G_file['G-TENSOR']
                    self._logger.message(f'G stored in file {G}')
                else:
                    # store on cpu because this can be large
                    self._G = G
                    self._logger.message('G stored in memory')
                self._logger.message(f'G shape = {self._G.shape}')
                G_count = torch.prod(torch.tensor(self._G.shape))
                bits = torch.finfo(torch_dtype).bits
                G_size_MB = (G_count * bits) / 8e6
                self._logger.message(f'G count = {G_count}')
                self._logger.message(f'G size (MB) = {G_size_MB}')

            with self._logger.subproc('Configuring batch and device'):
                # batch
                if batch_size is None:
                    batch_size = max(self._q_dims)
                self._batch_ids = _form_batch_ids(self._q_dims, batch_size)
                # device
                self._device = device
                self._logger.message(f'batch size = {batch_size}')
                self._logger.message(
                    f'batch count = {len(self._batch_ids)}')
                self._logger.message(f'computing device = {self._device}')

        ######################
        # uninitialized data #
        ######################
        # data
        self._nv = torch.tensor([], dtype=torch_dtype)
        self._mu = torch.tensor([], dtype=torch_dtype)
        self._sigma = torch.tensor([], dtype=torch_dtype)

        # auto scaling factor
        self._xi_mag = 1.
        self._b_mag = 1.

        # results saved during iterations
        self._saved_results = []
        self._save_iter = None

        # Jacobian and Hessian
        # NOTE: scipy calls jac() and hess() alternatively, so we can
        #       avoid calculating many repeated terms by forming both
        #       J and H when jac() is called
        self._jac_save = None
        self._hess_save = None
        self._jac_turn = True  # safety check
        self._x_save = None  # safety check

        # other options
        self._return_intensity_sensitivity = True

    def compute_intensity(self, w_dict, xi, b):
        """
        Compute intensity

        :param w_dict: `dict` of parameter weights
        :param xi: value of ξ
        :param b: value of b
        :return: intensity as a `torch.Tensor`
        """
        w_list = [w_dict[key].to(self._device) for key in self._par_keys]
        return self._intensity(w_list, xi, b).to('cpu')

    def solve_inverse(self, mu, sigma, nu_mu=.0, nu_sigma=1.,
                      w_dict_init=None, xi_init=None, b_init=None,
                      auto_scaling=True, maxiter=1000, verbose=1,
                      trust_options=None, save_iter=None,
                      return_intensity_sensitivity=True,
                      only_test_jac_hess=False):
        """
        Solve the inverse problem with observed intensity

        :param mu: mean (μ) of observed intensity
        :param sigma: stddev (σ) of observed intensity
        :param nu_mu: power of μ in χ2 normalisation (default=0)
        :param nu_sigma: power of σ in χ2 normalisation (default=1)
        :param w_dict_init: initial guess of weights (default=None)
        :param xi_init: initial guess of ξ (default=None)
        :param b_init: initial guess of b (default=None)
        :param auto_scaling: automatically scale data to proper magnitude
            to improve accuracy (default=True)
        :param maxiter: max number of iterations (default=1000)
        :param verbose: verbose level during inversion (default=1)
        :param trust_options: other options for
            `scipy.optimize.minimize(method='trust-constr')` (default=None)
        :param save_iter: save results every `save_iter` iterations
            (default=None)
        :param return_intensity_sensitivity: return intensity and sensitivity
            predicted by the inverted weights, ξ and b (default=True)
        :param only_test_jac_hess: only test Jacobian and Hessian
            (default=False)
        :return: a dict including the following (key, value) pairs:
            w_dict: `dict` of inverted weights
            xi: inverted ξ
            b: inverted b
            sens_w_dict: `dict` of normalized sensitivity of inverted weights
            sens_xi: normalized sensitivity of inverted ξ
            sens_b: normalized sensitivity of inverted b
            I: fitted intensity
            wct: wall-clock time
            opt_res: result of `scipy.optimize.minimize(method='trust-constr')`
            saved_res: `list` of `dict`'s, each `dict` containing the above
                (key, value) pairs saved every `save_iter` iterations
        """

        with self._logger.subproc('Solving inverse problem NLP-s'):

            with self._logger.subproc('Data processing'):
                with self._logger.subproc('Computing nu, d/nu and mu/nu'):
                    # ν=μ^nu_mu*σ^nu_sigma
                    self._nv = (mu ** nu_mu) * (sigma ** nu_sigma)
                    # send all to device
                    self._nv = _to_tensor(self._nv, device=self._device)
                    self._mu = _to_tensor(mu, device=self._device)
                    self._sigma = _to_tensor(sigma, device=self._device)
                    self._logger.message(
                        f'using nu=mu^{nu_mu}*sigma^{nu_sigma}')

                with self._logger.subproc('Determining xi0 and b0'):
                    if xi_init is None or b_init is None:
                        with self._logger.subproc('Computing mean intensity '
                                                  'over entire G'):
                            I_ave = torch.zeros_like(self._mu)
                            for G_id in self._batch_ids:
                                g = _to_tensor(self._G[G_id],
                                               device=self._device)
                                sum_g = torch.sum(g, dim=list(
                                    range(self._nq, self._nq + self._ns)))
                                I_ave[G_id] += sum_g / self._nv[G_id]
                            I_ave /= torch.prod(torch.tensor(self._s_dims))

                        with self._logger.subproc('Solving xi0 and b0'):
                            # min L, L = (xi I + b d - mu) ^ 2
                            mu_over_nv = self._mu / self._nv
                            d_over_nv = 1 / self._nv
                            a11 = (I_ave ** 2).sum()
                            a12 = (I_ave * d_over_nv).sum()
                            a22 = (d_over_nv ** 2).sum()
                            b1 = (mu_over_nv * I_ave).sum()
                            b2 = (mu_over_nv * d_over_nv).sum()
                            A = a11 * a22 - a12 * a12
                            xi0_auto = (b1 * a22 - b2 * a12) / A
                            b0_auto = (b2 * a11 - b1 * a12) / A
                    else:
                        xi0_auto = None
                        b0_auto = None

                    if xi_init is None:
                        xi0 = xi0_auto
                        self._logger.message(f'xi0 = {xi0.item()} '
                                             f'(determined by G)')
                    else:
                        xi0 = torch.tensor(xi_init, device=self._device)
                        self._logger.message(f'xi0 = {xi0.item()} '
                                             f'(specified by xi_init)')
                    if b_init is None:
                        b0 = b0_auto
                        self._logger.message(f'b0 = {b0.item()} '
                                             f'(determined by G)')
                    else:
                        b0 = torch.tensor(b_init, device=self._device)
                        self._logger.message(f'b0 = {b0.item()} '
                                             f'(specified by b_init)')

                with self._logger.subproc('Auto scaling'):
                    if auto_scaling:
                        # scaling for xi
                        self._xi_mag = _order_of_magnitude(xi0)
                        xi0 /= self._xi_mag
                        # scaling for b
                        self._b_mag = _order_of_magnitude(b0)
                        b0 /= self._b_mag
                        # log
                        self._logger.message(
                            f'scaling factor of xi = {self._xi_mag}')
                        self._logger.message(
                            f'scaling factor of b = {self._b_mag}')
                        self._logger.message(f'scaled xi0 = {xi0.item()}')
                        self._logger.message(f'scaled b0 = {b0.item()}')
                    else:
                        self._b_mag = 1.
                        self._xi_mag = 1.
                        self._logger.message(f'auto scaling disabled')

                # xi0, b0 to scalars
                xi0, b0 = xi0.item(), b0.item()

            # test Jacobian and Hessian
            if only_test_jac_hess:
                self._test_jac_hess(xi0 * 2, b0 / 2)
                return

            # other options
            self._return_intensity_sensitivity = return_intensity_sensitivity

            with self._logger.subproc('Solving NLP-s by trust-region'):
                with self._logger.subproc('Trust-region options'):
                    if trust_options is None:
                        trust_options = {}
                    trust_options['maxiter'] = maxiter
                    trust_options['verbose'] = verbose
                    self._logger.message(f'options = {trust_options}')

                with self._logger.subproc('Defining constrains |s|=1'):
                    s2 = optimize.NonlinearConstraint(self._constr_s2, 0., 0.,
                                                      jac=self._constr_s2_J,
                                                      hess=self._constr_s2_H)

                with self._logger.subproc(
                        'Running '
                        'scipy.optimize.minimize(method="trust-constr")'):
                    # prepare saved results
                    self._save_iter = save_iter
                    self._saved_results = []
                    if save_iter is not None:
                        self._logger.message(f'saving results every '
                                             f'{save_iter} iterations')
                    # redirect stdout from scipy
                    stdout_default = sys.stdout
                    sys.stdout = self._logger.get_writer()
                    try:
                        opt_res = optimize.minimize(
                            self._obj_func,
                            self._get_x0(xi0, b0, w_dict_init=w_dict_init),
                            method='trust-constr',
                            jac=self._jac_func,
                            hess=self._hess_func,
                            constraints=(s2,),
                            options=trust_options,
                            callback=self._call_back_save)
                    except Exception as err:
                        raise err
                    finally:
                        # restore stdout
                        sys.stdout = stdout_default

                with self._logger.subproc('Unpacking results'):
                    result_dict = self._unpack_result(opt_res)
                    result_dict['saved_res'] = self._saved_results
                    return result_dict
