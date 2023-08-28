# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 09:56:43 2023

@author: hojun
"""

import torch
import torch.nn as nn


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        print(m)
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)