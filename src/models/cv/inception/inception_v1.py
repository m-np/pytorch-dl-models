"""
model - Inception V1 
reference : https://arxiv.org/abs/1409.4842

This model was developed for image classification task on ImageNet Dataset
"""
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
            "in_channels" : 3,
            "num_classes": 1000,
            "layer_filter": 64,

            "conv1_filters": 64,
            "conv2_filters": 128,
            "conv3_filters": 256,
            "conv4_filters": 512,

            "expansion": 4,
            "num_blocks_1": 3,
            "num_blocks_2": 4,
            "num_blocks_3": 6,
            "num_blocks_4": 3,
            }
    return params

class ConvBlock(nn.Module):
    """
    Convolution block consists of 
    1. Convolution 2d
    2. BatchNormalization
    3. Relu
    """
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size,
            stride,
            padding,
            bias = False):
        super().__init__()

        self.conv = nn.Sequential(
                        nn.Conv2d(
                            in_channels = in_channels, 
                            out_channels = out_channels, 
                            kernel_size = kernel_size, 
                            stride = stride, 
                            padding = padding,
                            bias = bias),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        )

    def forward(self, x):
        return self.conv(x)
    

class InceptionBlock(nn.Module):
    """
    Inception block is a combination of 4 individual blocks which are concated together
    1. 1 x 1 convolution
    2. 1 x 1 convolution followed by 3 x 3
    3. 1 x 1 convolution followed by 5 x 5
    4. Maxpooling followed by 1 x 1

    To generate same height and width of output feature map as the input feature map, following should be padding for
        * 1x1 conv : p=0
        * 3x3 conv : p=1
        * 5x5 conv : p=2
    """

    def __init__(
            self,
            in_channels,
            out1,
            out21,
            out22,
            out31,
            out32,
            out4,
            ):
        """
        Input Args:
        in_channels = Input channels
        out1 = Filter size for 1x1 conv in branch 1
        out21 = Filter size for 3x3 conv in branch 2
        out22 = Filter size for 1x1 conv in branch 2
        out31 = Filter size for 5x5 conv in branch 3
        out32 = Filter size for 1x1 conv in branch 3
        out4 = Filter size for 1x1 conv in branch 4
        """
        super().__init__()

        # branch1 : k=1,s=1,p=0
        self.branch1 = ConvBlock(
                        in_channels = in_channels, 
                        out_channels = out1, 
                        kernel_size = 1, 
                        stride = 1, 
                        padding= 0)
        
        # branch2 : k=1,s=1,p=0 -> k=3,s=1,p=1
        self.branch2 = nn.Sequential(
                        ConvBlock(
                            in_channels = in_channels, 
                            out_channels = out21, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding= 0),
                        ConvBlock(
                            in_channels = in_channels, 
                            out_channels = out22, 
                            kernel_size = 3, 
                            stride = 3, 
                            padding= 1)
                        )
        
        # branch3 : k=1,s=1,p=0 -> k=5,s=1,p=2
        self.branch3 = nn.Sequential(
                        ConvBlock(
                            in_channels = in_channels, 
                            out_channels = out31, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding= 0),
                        ConvBlock(
                            in_channels = in_channels, 
                            out_channels = out32, 
                            kernel_size = 5, 
                            stride = 5, 
                            padding= 2)
                        )
        
        # branch4 : pool(k=3,s=1,p=1) -> k=1,s=1,p=0
        self.branch4 = nn.Sequential(
                        nn.MaxPool2d(
                            kernel_size = 3, 
                            stride = 1,
                            padding = 1),
                        ConvBlock(
                            in_channels = in_channels, 
                            out_channels = out4, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding= 0)
                        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class Inception_v1(nn.Module):
    """
    Consists of 4 resid blocks and FC layer at the end
    """
    def __init__(self, params, device="cpu"):
        super().__init__()

        in_channels = params["in_channels"]
        num_classes = params["num_classes"]
        layer_filter = params["layer_filter"]
        expansion = params["expansion"]

        num_blocks_1 = params["num_blocks_1"]
        num_blocks_2 = params["num_blocks_2"]
        num_blocks_3 = params["num_blocks_3"]
        num_blocks_4 = params["num_blocks_4"]

        conv1_filters = params["conv1_filters"]
        conv2_filters = params["conv2_filters"]
        conv3_filters = params["conv3_filters"]
        conv4_filters = params["conv4_filters"]




        self.layer1 = nn.Sequential(
                        nn.Conv2d(
                            in_channels = in_channels, 
                            out_channels = layer_filter, 
                            kernel_size = 7, 
                            stride = 2, 
                            padding = 3),
                        nn.BatchNorm2d(layer_filter),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 3, 
                            stride = 2,
                            padding = 1)
                        )

        self.resid_conv1 = self.create_resid_layers(
                            in_channels = layer_filter, 
                            out_channels = conv1_filters, 
                            stride = 1, 
                            expansion = expansion, 
                            num_blocks = num_blocks_1,
                        )
        
        self.resid_conv2 = self.create_resid_layers(
                            in_channels = conv1_filters*expansion, 
                            out_channels = conv2_filters, 
                            stride = 2, 
                            expansion = expansion, 
                            num_blocks = num_blocks_2,
                        )
        
        self.resid_conv3 = self.create_resid_layers(
                            in_channels = conv2_filters*expansion, 
                            out_channels = conv3_filters, 
                            stride = 2, 
                            expansion = expansion, 
                            num_blocks = num_blocks_3,
                        )

        self.resid_conv4 = self.create_resid_layers(
                            in_channels = conv3_filters*expansion, 
                            out_channels = conv4_filters, 
                            stride = 2, 
                            expansion = expansion, 
                            num_blocks = num_blocks_4,
                        )
        
        # Average pooling (used in classification head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification task 
        # in_channel = 4, By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
        # Check line 77
        self.linear = nn.Linear(
                            in_features=conv4_filters*expansion, 
                            out_features=num_classes)

        
    def forward(self, x):
        # Perform convolutions
        x = self.layer1(x)
        C1 = self.resid_conv1(x)
        C2 = self.resid_conv2(x)
        C3 = self.resid_conv3(x)
        C4 = self.resid_conv4(x)

        out = self.avgpool(C4)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return C1, C2, C3, C4, out

