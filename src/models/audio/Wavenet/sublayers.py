# importing required libraries
import math

# torch packages
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation,
                bias = True,
                device = None,
                dtype = None
                ):
        """
        Input Args as per Conv1D documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        super().__init__()

        # Attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # Padding calculation when dilation is used
        self.padding = (kernel_size-1) * dilation
        self.dilation = dilation
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.conv = nn.Conv1d(
                            in_channels = self.in_channels,
                            out_channels = self.out_channels,
                            kernel_size = self.kernel_size,
                            stride = self.stride,
                            padding = self.padding,
                            dilation = self.dilation,
                            bias = self.bias,
                            device = self.device,
                            dtype = self.dtype
                            )
        
    def forward(self, x):
        """
        x -> (Batch_size, N_channels, seq_length)
        """
        # Make sure to remove k-1 featurees in the end as this will ensure that 
        # we don't use future result in our calculation for present state
        return self.conv(x[0: -(self.kernel_size - 1)])
    



        
