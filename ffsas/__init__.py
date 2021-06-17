#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# __init__.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" ffsas module """

import torch

# torch data type
# todo: ffsas supports float32 but scipy.optimize does not, so we
#       currently fix dtype to float64
torch_dtype = torch.float64

# specify version here; it will picked up by pip
__version__ = '1.0.0'
