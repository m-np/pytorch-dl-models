"""
Wavenet paper - https://arxiv.org/abs/1609.03499
"""
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
                kernel_size=1,
                stride=1,
                dilation=1,
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
        x = self.conv(x)
        x = x[:, :, 0: -(self.kernel_size - 1)]
        return x
    

class WaveBlock(nn.Module):
    """
    This block consists of Residual Convolution with Gated linear unit

    Reference : https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/modules.py
    """
    def __init__(
            self, 
            in_channels, 
            gate_channels, 
            kernel_size,
            stride, 
            dilation,
            skip_channels = None,
            local_conditioning_channels = -1,
            global_conditioning_channels = -1,
            pdropout = 0.1,
            bias = True,
            ):
        """
        Input Args:
        in_channels: Input/output channel to the unit
        gate_channels: Gated activation channel
        kernel_size: Size of kernels of convolution layer
        stride: Strides of convolution window 
        dilation: Dilation factor
        skip_channels: Skip connections channel. set to in_channels if None
        local_conditioning_channels: Input local conditioning channel, if -ve then its disabled
        global_conditioning_channels: Input global conditioning channel, if -ve then its disabled
        pdropout: Dropout Probability
        bias = True,
        """
        super(WaveBlock, self).__init__()

        self.dilation = dilation

        self.dropout = nn.Dropout(p=pdropout)

        if skip_channels is None:
            skip_channels = in_channels

        self.conv = CausalConv1d(
                            in_channels = in_channels, 
                            out_channels = gate_channels, 
                            kernel_size = kernel_size, 
                            stride = stride, 
                            dilation = dilation,
                            bias = bias
                            )
        
        # Page 5 of the wavenet paper, chapter 2.5
        # Local conditioning
        self.conv_local = None
        if local_conditioning_channels > 0:
            self.conv_local = nn.Conv1d(
                                    in_channels = local_conditioning_channels,
                                    out_channels = gate_channels,
                                    bias = False,
                                    )
        
        # Page 4 of the wavenet paper, chapter 2.5
        # Global conditioning
        self.conv_global = None
        if self.global_conditioning_channels > 0:
            self.conv_global = nn.Conv1d(
                                    in_channels = global_conditioning_channels,
                                    out_channels = gate_channels,
                                    bias = False,
                                    )

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2

        self.conv_out = nn.Conv1d(
                            in_channels = gate_out_channels,
                            out_channels = in_channels,
                            bias = bias,
                            )

        self.conv_skip = nn.Conv1d(
                            in_channels = gate_out_channels,
                            out_channels = skip_channels,
                            bias = bias,
                            )
        
    def forward(self, x, c = None, g = None):
        """
        Args: 
        x = Input feature -> Batch_size, Channel, Seq_len
        # reference page 4 and 5 of Wavenet paper, chapter 2.5
        c = local conditioning feature -> Batch_size, Channel, Seq_len
        g = global conditioning feature -> Batch_size, Channel, Seq_len
        """
        residual = x
        x = self.dropout(x)
        # Apply causal convolution
        x = self.conv(x)
        # Split the features for gated input on the channel dimensions
        splitdim = 1
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        # Apply local conditioning
        if c is not None:
            assert self.conv_local is not None, \
            f"Initialize local_conditioning_channels to apply local conditioning"

            c = self.conv_local(c)
            ca, cb = x.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # Apply global conditioning
        if g is not None:
            assert self.conv_global is not None, \
            f"Initialize global_conditioning_channels to apply global conditioning"

            g = self.conv_global(g)
            ga, gb = x.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb
        
        x = torch.tanh(a) * torch.sigmoid(b)

        # Apply skip connection
        s = self.conv_skip(x)

        # Apply residual connection
        x = self.conv_out(x)
        x = (x + residual) * math.sqrt(0.5)

        return x, s






        
