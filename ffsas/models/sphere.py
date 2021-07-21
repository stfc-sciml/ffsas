#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# sphere.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" sphere model class """

import math

import torch

from ffsas.models.base import SASModel


class Sphere(SASModel):
    @classmethod
    def compute_G(cls, q_list, par_dict, const_dict, V=None):
        # get parameters
        q = q_list[0]
        r = par_dict['r']
        drho = const_dict['drho']

        # compute volume
        if V is None:
            V = cls.compute_V(par_dict)

        #############
        # Compute G #
        #############

        # step 1: qr
        qr = torch.outer(q, r)

        # step 2: shape factor
        shape_factor = 3. * (torch.sin(qr) - qr * torch.cos(qr)) / qr ** 3
        # limit at 0
        shape_factor = torch.nan_to_num(shape_factor,
                                        nan=1., posinf=1., neginf=1.)

        # step 3: G
        G = (drho * V[None, :] * shape_factor) ** 2
        return G

    @classmethod
    def get_par_keys_G(cls):
        return ['r']

    @classmethod
    def compute_V(cls, par_dict):
        r = par_dict['r']
        return 4. / 3. * math.pi * r ** 3

    @classmethod
    def get_par_keys_V(cls):
        return ['r']
