"""
ALexnet model -
reference : https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5

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
            "num_classes": 1000,
            "linear_out": 4096,
            "conv1_filters": 96,
            "conv2_filters": 256,
            "conv3_filters": 384,
            "conv4_filters": 384,
            "conv5_filters": 256,
            "dropout_probability": 0.5,
            }
    return params


class AlexNet(nn.Module):
    """
    Consists of 2 blocks

    1. Convolution block 5 layers containig
        - convolution2d
        - batchnorm2d
        - relu
        - maxpooling2d
    2. Fully Connected Layers 3 block containing
        - dropout
        - linear
        - relu
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
        dropout_probability = params["dropout_probability"]

        self.layer1 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=3, 
                            out_channels=conv1_filters, 
                            kernel_size=11, 
                            stride=4, 
                            padding=0),
                        nn.BatchNorm2d(conv1_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 3, 
                            stride = 2)
                        )
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=conv1_filters, 
                            out_channels=conv2_filters, 
                            kernel_size=5, 
                            stride=1, 
                            padding=2),
                        nn.BatchNorm2d(conv2_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(
                            kernel_size = 3, 
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
                        nn.MaxPool2d(
                            kernel_size = 3, 
                            stride = 2)
                        )
        
        self.fc1 = nn.Sequential(
                        nn.Dropout(dropout_probability),
                        nn.Linear(
                            in_features=9216, 
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
        x = self.avgpool1(self.tanh1(self.conv1(x)))
        x = self.avgpool2(self.tanh2(self.conv2(x)))
        x = self.flatten(x)
        x = self.tanh3(self.linear1(x))
        x = self.tanh4(self.linear2(x))
        x = self.linear3(x)
        return x


        



