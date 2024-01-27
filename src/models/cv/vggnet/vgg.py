"""
Model - Vgg16

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
            "num_classes": 10,
            "linear_out": 4096,
            "conv1_filters": 64,
            "conv2_filters": 64,
            "conv3_filters": 128,
            "conv4_filters": 128,
            "conv5_filters": 256,
            "conv6_filters": 256,
            "conv7_filters": 256,
            "conv8_filters": 512,
            "conv9_filters": 512,
            "conv10_filters": 512,
            "conv11_filters": 512,
            "conv12_filters": 512,
            "conv13_filters": 512,
            "dropout_probability": 0.5,
            }
    return params


class VGG16(nn.Module):
    """
    VGG16 is a 16 layered model 
    with 13 blocks of convolution and 3 blocks of FC
    1. Block Convolution
        - Convolution2d
        - BatchNorm2d
        - Relu
        - MaxPool2d (in some layers)
    """
    def __init__(self, params, device="cpu"):
        super().__init__()

        num_classes = params["num_classes"]
        linear_out = params["linear_out"]
        conv1_filters = params["conv1_filters"]
        conv2_filters = params["conv2_filters"]
        conv3_filters = params["conv3_filters"]
        conv4_filters = params["conv4_filters"]
        conv5_filters = params["conv5_filters"]
        conv6_filters = params["conv6_filters"]
        conv7_filters = params["conv7_filters"]
        conv8_filters = params["conv8_filters"]
        conv9_filters = params["conv9_filters"]
        conv10_filters = params["conv10_filters"]
        conv11_filters = params["conv11_filters"]
        conv12_filters = params["conv12_filters"]
        conv13_filters = params["conv13_filters"]
        dropout_probability = params["dropout_probability"]


        self.layer1 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=3, 
                            out_channels=conv1_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv1_filters),
                        nn.ReLU(),
                        )
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv1_filters, 
                            out_channels=conv2_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv2_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 2, 
                            stride = 2)
                        )
        
        self.layer3 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv2_filters, 
                            out_channels=conv3_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv3_filters),
                        nn.ReLU(),
                        )
        
        self.layer4 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv3_filters, 
                            out_channels=conv4_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv4_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 2, 
                            stride = 2)
                        )
        
        self.layer5 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv4_filters, 
                            out_channels=conv5_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv5_filters),
                        nn.ReLU(),
                        )
        
        self.layer6 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv5_filters, 
                            out_channels=conv6_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv6_filters),
                        nn.ReLU(),
                        )
        
        self.layer7 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv6_filters, 
                            out_channels=conv7_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv7_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 2, 
                            stride = 2)
                        )
        
        self.layer8 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv7_filters, 
                            out_channels=conv8_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv8_filters),
                        nn.ReLU(),
                        )
        
        self.layer9 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv8_filters, 
                            out_channels=conv9_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv9_filters),
                        nn.ReLU(),
                        )
        
        self.layer10 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv9_filters, 
                            out_channels=conv10_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv10_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 2, 
                            stride = 2)
                        )
        
        self.layer11 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv10_filters, 
                            out_channels=conv11_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv11_filters),
                        nn.ReLU(),
                        )
        
        self.layer12 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv11_filters, 
                            out_channels=conv12_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv12_filters),
                        nn.ReLU(),
                        )
        
        self.layer13 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv12_filters, 
                            out_channels=conv13_filters, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
                        nn.BatchNorm2d(conv13_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 2, 
                            stride = 2)
                        )

        self.fc1 = nn.Sequential(
                        nn.Dropout(dropout_probability),
                        nn.Linear(
                            in_features=7*7*512, 
                            out_features=linear_out),
                        nn.ReLU()
                        )
        
        self.fc2 = nn.Sequential(
                        nn.Dropout(dropout_probability),
                        nn.Linear(
                            in_features=linear_out, 
                            out_features=linear_out),
                        nn.ReLU()
                        )
        
        self.fc3 = nn.Sequential(
                        nn.Dropout(dropout_probability),
                        nn.Linear(
                            in_features=linear_out, 
                            out_features=num_classes),
                        nn.ReLU()
                        )
        
    def forward(self, x):
        # Perform convolutions
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


        



