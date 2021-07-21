#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ellipsoid.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" ellipsoid model class """

import math

import torch

from ffsas.models.base import SASModel


class Ellipsoid(SASModel):
    @classmethod
    def compute_G(cls, q_list, par_dict, const_dict, V=None):
        # get parameters
        qx, qy = q_list[0], q_list[1]
        rp, re = par_dict['rp'], par_dict['re']
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

        # step 2: qr
        qa_re = torch.moveaxis(qa[:, :, :, :, None] *
                               re[None, None, None, None, :], 4, 2)
        qc_rp = torch.moveaxis(qc[:, :, :, :, None] *
                               rp[None, None, None, None, :], 4, 2)
        qr = torch.sqrt(torch.square(qa_re)[:, :, None, :, :, :] +
                        torch.square(qc_rp)[:, :, :, None, :, :])

        # step 3: shape factor
        shape_factor = 3. * (torch.sin(qr) - qr * torch.cos(qr)) / qr ** 3
        # limit at 0
        shape_factor = torch.nan_to_num(shape_factor,
                                        nan=1., posinf=1., neginf=1.)

        # step 4: G
        G = (drho * shape_factor * V[None, None, :, :, None, None]) ** 2
        return G

    @classmethod
    def get_par_keys_G(cls):
        return ['rp', 're', 'theta', 'phi']

    @classmethod
    def compute_V(cls, par_dict):
        rp, re = par_dict['rp'], par_dict['re']
        return 4. / 3. * math.pi * rp[:, None] * re[None, :] ** 2

    @classmethod
    def get_par_keys_V(cls):
        return ['rp', 're']
