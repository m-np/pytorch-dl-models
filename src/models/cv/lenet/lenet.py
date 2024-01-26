import math
import copy
import time
import random
import spacy
import numpy as np
import os 

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim


def get_params():
    params = {
            "linear_out1": 120,
            "linear_out2": 84,
            "linear_out3": 10,
            "conv1_filters": 6,
            "conv2_filters": 16,
            "kernel_size": 5,
            "padding": 2,
            }
    return params


class LeNet(nn.Module):
    """
    Consists of 2 blocks

    1. Convolution block 2 layers containig
        - convolution2d
        - tanh
        - avgpooling2d
    2. Fully Connected Layers 2 block containing and a flattern layer
        - linear
        - tanh
    """
    def __init__(self, params, device="cpu"):
        super().__init__()

        linear_out1 = params["linear_out1"]
        linear_out2 = params["linear_out2"]
        linear_out3 = params["linear_out3"]
        conv1_filters = params["conv1_filters"]
        conv2_filters = params["conv2_filters"]
        kernel_size = params["kernel_size"]
        padding = params["padding"]

        self.conv1 = nn.Conv2d(
                            in_channels=1, 
                            out_channels=conv1_filters, 
                            kernel_size=kernel_size, 
                            stride=1, 
                            padding=padding)  # B x 28 x 28
        self.tanh1 = nn.Tanh()
        self.avgpool1 = nn.AvgPool2d(
                            kernel_size=2, 
                            stride=2)   # B x 14 x 14
        
        self.conv2 = nn.Conv2d(
                            in_channels=conv1_filters, 
                            out_channels=conv2_filters, 
                            kernel_size=kernel_size, 
                            stride=1, 
                            padding=padding)  # B x 10 x 10
        self.tanh2 = nn.Tanh()
        self.avgpool2 = nn.AvgPool2d(
                            kernel_size=2, 
                            stride=2)   # B x 5 x 5
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
                            in_features=16*5*5, 
                            out_features=linear_out1)
        self.tanh3 = nn.Tanh()
        self.linear2 = nn.Linear(
                            in_features=linear_out1, 
                            out_features=linear_out2)
        self.tanh4 = nn.Tanh()
        self.linear3 = nn.Linear(
                            in_features=linear_out2, 
                            out_features=linear_out3)
        
    def forward(self, x):
        # Perform convolutions
        x = self.avgpool1(self.tanh1(self.conv1(x)))
        x = self.avgpool2(self.tanh2(self.conv2(x)))
        x = self.flatten(x)
        x = self.tanh3(self.linear1(x))
        x = self.tanh4(self.linear2(x))
        x = self.linear3(x)
        return x


        



