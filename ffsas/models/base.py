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
from ffsas.utils import _form_batch_ids, MultiLevelLogger


class SASModel:
    @classmethod
    @abstractmethod
    def compute_G(cls, q_list, par_dict, const_dict, V=None):
        """
        Compute the Green's tensor `G`

        :param q_list: `list` of scattering vectors `q`
        :param par_dict: `dict` of model parameters as `torch.Tensor`s
        :param const_dict: `dict` of model constants
        :param V: the volume tensor (default=`None`, compute if needed)
        :return: the Green's tensor `G` as a `torch.Tensor`
        """
        pass

    @classmethod
    @abstractmethod
    def get_par_keys_G(cls):
        """
        Get the keys of the parameters used for computing `G`

        :return: `list` of parameters keys ordered by their dimensions in `G`
        """
        pass

    @classmethod
    @abstractmethod
    def compute_V(cls, par_dict):
        """
        Compute the volume `V`

        :param par_dict: `dict` of model parameters as `torch.Tensor`s
        :return: `V` as a `torch.Tensor`
        """
        pass

    @classmethod
    @abstractmethod
    def get_par_keys_V(cls):
        """
        Get the keys of the parameters used for computing `V`

        :return: `list` of parameters keys ordered by their dimensions in `V`
        """
        pass

    @classmethod
    def compute_G_mini_batch(cls, q_list, par_dict, const_dict,
                             fixed_par_weights=None,
                             G_file=None, batch_size=None, device='cpu',
                             log_file=None, log_screen=True):
        """
        Compute the Green's tensor `G` by GPU-accelerated mini-batch computation

        :param q_list: `list` of scattering vectors `q`
        :param par_dict: `dict` of model parameters as `torch.Tensor`s
        :param const_dict: `dict` of model constants
        :param fixed_par_weights: `dict` of fixed parameters and their weights
        :param G_file: write `G` to a HDF5 file named `G_file`, with the
            dateset named `G-TENSOR` (default=None, return `G` as a
            `torch.Tensor`)
        :param batch_size: batch size along each `q`-dimension
            (default=None, compute entire `G` in one batch)
        :param device: computing device (default='cpu')
        :param log_file: save log to this file (default=None)
        :param log_screen: print log to stdout (default=True)
        :return: if `G_file` is `None`, return `G` as a `torch.Tensor`;
            otherwise, return `G_file` as a string
        """
        # logger
        logger = MultiLevelLogger()
        logger.activate(file_path=log_file, screen=log_screen)

        with logger.subproc("Computing the Green's tensor G"):
            logger.message(f'Class name = {cls.__name__}')

            with logger.subproc("Checking tensor dtypes"):
                for q in q_list:
                    assert q.dtype == torch_dtype, \
                        f'dtype of q must be {torch_dtype}'
                for key, par in par_dict.items():
                    assert par.dtype == torch_dtype, \
                        f'dtype of parameter "{key}" must be {torch_dtype}'
                logger.message(f'torch dtype = {torch_dtype}')

            with logger.subproc("Computing volume V"):
                V = cls.compute_V(par_dict)

            with logger.subproc("Handling fixed parameters"):
                if fixed_par_weights is None:
                    fixed_par_weights = {}
                par_keys_free = []
                par_dims_fixed = []
                for loc, key in enumerate(cls.get_par_keys_G()):
                    if key not in fixed_par_weights.keys():
                        par_keys_free.append(key)
                    else:
                        par_dims_fixed.append(loc)
                        assert fixed_par_weights[key].size(0) == par_dict[
                            key].size(0), \
                            f'sizes of weights and coordinates do not match ' \
                            f'for fix parameter "{key}"'
                logger.message(f'model parameters = {cls.get_par_keys_G()}')
                logger.message(
                    f'fixed parameters = {list(fixed_par_weights.keys())}')
                logger.message(f'free parameters = {par_keys_free}')

            with logger.subproc("Creating space for G"):
                # container for G
                q_dims = [q.size(0) for q in q_list]
                G_shape = q_dims + [par_dict[key].size(0) for key in
                                    par_keys_free]
                logger.message(f'G shape = {G_shape}')
                G_count = torch.prod(torch.tensor(G_shape))
                bits = torch.finfo(torch_dtype).bits
                G_size_MB = (G_count * bits) / 8e6
                logger.message(f'G count = {G_count}')
                logger.message(f'G size (MB) = {G_size_MB}')
                if G_file is None:
                    G = torch.empty(G_shape, dtype=q_list[0].dtype)
                    G_h5 = False
                    logger.message('G stored in memory')
                else:
                    os.system(f'rm -f {G_file}')
                    G_h5 = h5py.File(G_file, 'w')
                    G = G_h5.create_dataset('G-TENSOR', G_shape,
                                            dtype=str(q_list[0].dtype)[6:])
                    logger.message(f'G stored in file {G_file}')

            with logger.subproc("Sending parameters and constants to device"):
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

                # sending fixed weights to device
                fixed_par_weights_device = {}
                for key, weights in fixed_par_weights.items():
                    fixed_par_weights_device[key] = weights.to(device)

                logger.message(f'device = {device}')

            with logger.subproc("Creating batch indices"):
                # create batch ids
                if batch_size is None:
                    batch_size = max(q_dims)
                batch_ids = _form_batch_ids(q_dims, batch_size)
                logger.message(f'batch size = {batch_size}')
                logger.message(f'batch count = {len(batch_ids)}')

            with logger.subproc("Computing G by batches"):
                for i, batch_id in enumerate(batch_ids):
                    q_list_device = []
                    for iq, q in enumerate(q_list):
                        q_list_device.append(q[batch_id[iq]].to(device))
                    batch = cls.compute_G(q_list_device, par_dict_device,
                                          const_dict_device, V=V.to(device))
                    # fixed parameters
                    for dim in par_dims_fixed[::-1]:
                        w = fixed_par_weights_device[cls.get_par_keys_G()[dim]]
                        batch = torch.tensordot(
                            batch, w, dims=[[len(q_dims) + dim], [0]])
                    G[tuple(batch_id)] = batch.to('cpu')
                    logger.message(f'batch {i + 1} / {len(batch_ids)}, '
                                   f'elapsed={logger.elapsed_shallowest:f} sec')

            # return
            if G_h5:
                G_h5.close()
                return G_file
            else:
                return G

    @classmethod
    def compute_average_V(cls, par_dict, w_dict):
        """
        Compute average volume

        :param par_dict: `dict` of model parameters as `torch.Tensor`s
        :param w_dict: `dict` of parameter weights as `torch.Tensor`s
        :return: average volume
        """
        V = cls.compute_V(par_dict)
        for i, key in enumerate(cls.get_par_keys_V()[::-1]):
            V = torch.tensordot(V, w_dict[key], dims=1)
        return V.item()
