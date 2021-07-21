#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# cylinder.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" cylinder model class """

import math

import torch
from scipy.special import j1

from ffsas.models.base import SASModel


class Cylinder(SASModel):
    @classmethod
    def compute_G(cls, q_list, par_dict, const_dict, V=None):
        # get parameters
        qx, qy = q_list[0], q_list[1]
        l, r = par_dict['l'], par_dict['r']
        theta, phi = par_dict['theta'], par_dict['phi']
        drho = const_dict['drho']

        # compute volume
        if V is None:
            V = cls.compute_V(par_dict)

        #############
        # Compute G #
        #############

        # step 1: rotate q
        sin_theta = torch.sin(theta)
        r31 = torch.outer(sin_theta, torch.cos(phi))
        r32 = torch.outer(sin_theta, torch.sin(phi))
        qc = (qx[:, None, None] * r31[None, :, :])[:, None, :, :] + \
             (qy[:, None, None] * r32[None, :, :])[None, :, :, :]
        qa = torch.sqrt(torch.clip(
            (qx ** 2)[:, None, None, None] +
            (qy ** 2)[None, :, None, None] - qc ** 2, min=0.))

        # step 2: qa * r, qc * l
        qa_r = torch.moveaxis(qa[:, :, :, :, None] *
                              r[None, None, None, None, :], 4, 2)
        qc_l = torch.moveaxis(qc[:, :, :, :, None] *
                              l[None, None, None, None, :], 4, 2) * .5

        # step 3: shape factor
        sin_qc_l = torch.nan_to_num(2. * torch.sin(qc_l) / qc_l,
                                    nan=1., posinf=1., neginf=1.)
        # NOTE: scipy.special.j1() must be called on cpu
        j1_qa_r = torch.tensor(j1(qa_r.to('cpu').numpy()), device=qa_r.device)
        j1_qa_r = torch.nan_to_num(j1_qa_r / qa_r, nan=1., posinf=1., neginf=1.)
        # shape factor
        shape_factor = \
            sin_qc_l[:, :, :, None, :, :] * j1_qa_r[:, :, None, :, :, :]

        # step 4: G
        G = (drho * shape_factor * V[None, None, :, :, None, None]) ** 2
        return G

    @classmethod
    def get_par_keys_G(cls):
        return ['l', 'r', 'theta', 'phi']

    @classmethod
    def compute_V(cls, par_dict):
        l, r = par_dict['l'], par_dict['r']
        return math.pi * l[:, None] * r[None, :] ** 2

    @classmethod
    def get_par_keys_V(cls):
        return ['l', 'r']
