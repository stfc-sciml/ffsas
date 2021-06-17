#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# base.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" base class SASModel """

import os
from abc import abstractmethod

import h5py
import torch

from ffsas import torch_dtype
from ffsas.utils import _form_batch_ids


class SASModel:
    @classmethod
    @abstractmethod
    def compute_G(cls, q_list, par_dict, const_dict, V=None):
        """
        Compute the Green's tensor `G`

        :param q_list: a list of scattering vectors `q`
        :param par_dict: a dict of model parameters as `torch.Tensor`s
        :param const_dict: a dict of model constants as scalars
        :param V: the volume tensor (default=`None`, compute it if needed)
        :return: `G` as a `torch.Tensor`
        """
        pass

    @classmethod
    @abstractmethod
    def get_par_keys_G(cls):
        """
        Get the keys of the parameters used for computing `G`

        :return: a list of parameters keys ordered by their dimensions in `G`
        """
        pass

    @classmethod
    @abstractmethod
    def compute_V(cls, par_dict, const_dict):
        """
        Compute the volume `V`

        :param par_dict: a dict of model parameters as `torch.Tensor`s
        :param const_dict: a dict of model constants as scalars
        :return: `V` as a `torch.Tensor`
        """
        pass

    @classmethod
    @abstractmethod
    def get_par_keys_V(cls):
        """
        Get the keys of the parameters used for computing `V`

        :return: a list of parameters keys ordered by their dimensions in `V`
        """
        pass

    @classmethod
    def compute_G_V(cls, q_list, par_dict, const_dict,
                    G_file=None, batch_size=None, device='cpu'):
        """
        Compute the Green's tensor `G` and volume `V` for class `model` and
        the given parameters in `q_list`, `par_dict` and `const_dict`

        :param q_list: a list of scattering vectors `q`
        :param par_dict: a dict of model parameters as `torch.Tensor`s
        :param const_dict: a dict of model constants as scalars
        :param G_file: write `G` to a HDF5 file named `G_file`, with the
            dateset named `G-TENSOR` (default=None, return `G` as a
            `torch.Tensor`)
        :param batch_size: batch size along each `q`-dimension for
            mini-batch computation of G-based terms (default=None, no
            mini-batch computation)
        :param device: computing device (default='cpu')
        :return: `G` and `V`; if `G_file` is `None`, `G` will be a
            `torch.Tensor`; otherwise `G` will be the filename or `G_file`
        """
        # check dtype
        for q in q_list:
            assert q.dtype == torch_dtype, \
                f'dtype of q must be {torch_dtype}'
        for key, par in par_dict.items():
            assert par.dtype == torch_dtype, \
                f'dtype of parameter "{key}" must be {torch_dtype}'

        # volume (on CPU, should be fast)
        V = cls.compute_V(par_dict, const_dict)

        # container for G
        q_dims = [q.size(0) for q in q_list]
        G_shape = q_dims + [par_dict[key].size(0) for key in
                            cls.get_par_keys_G()]
        if G_file is None:
            # in-memory tensor on cpu
            G = torch.zeros(G_shape, dtype=q_list[0].dtype)
            G_h5 = False
        else:
            # save to hdf5
            os.system(f'rm -f {G_file}')
            G_h5 = h5py.File(G_file, 'w')
            G = G_h5.create_dataset('G-TENSOR', G_shape,
                                    dtype=str(q_list[0].dtype)[6:])

        # send par_dict to device
        par_dict_device = {}
        for key, par in par_dict.items():
            par_dict_device[key] = par.to(device)

        # send tensors in const_dict to device
        const_dict_device = {}
        for key, const in const_dict.items():
            if isinstance(const, torch.Tensor):
                const_dict_device[key] = const.to(device)
            else:
                const_dict_device[key] = const

        # create batch ids
        if batch_size is None:
            batch_size = max(q_dims)
        batch_ids = _form_batch_ids(q_dims, batch_size)

        # compute G on batches
        for batch_id in batch_ids:
            q_list_device = []
            for iq, q in enumerate(q_list):
                q_list_device.append(q[batch_id[iq]].to(device))
            batch = cls.compute_G(q_list_device, par_dict_device,
                                  const_dict_device, V=V.to(device))
            G[tuple(batch_id)] = batch.to('cpu')

        # return
        if G_h5:
            G_h5.close()
            return G_file, V
        else:
            return G, V
