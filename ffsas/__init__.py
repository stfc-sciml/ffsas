#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# __init__.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" ffsas module """

import torch

# data type
torch_dtype = torch.float64
torch.set_default_dtype(torch_dtype)


def set_torch_dtype(dtype):
    """ set torch dtype """
    global torch_dtype
    torch_dtype = dtype
    torch.set_default_dtype(torch_dtype)


# specify version here; it will picked up by pip
__version__ = '1.0.1'
