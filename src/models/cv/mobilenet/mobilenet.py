"""
model - MobileNetV1
reference : https://arxiv.org/abs/1704.04861

This model was developed for image classification task for CIFAR-10..
This model was designed for speed and not size as they use depthwise convolution to build a light weight deep NN
"""

from collections import OrderedDict

# torch packages
import torch
import torch.nn as nn


def get_params():
    params = {
        "in_channels": 3,
        "num_classes": 1000,
        "channels": [32, 64, 128, 256, 512, 1024],
        "width_multiplier": 1,
    }
    return params


class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels, out_channels=in_channels, stride=stride
            ),
            PointwiseConv(in_channels=in_channels, out_channels=out_channels, stride=1),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNet(nn.Module):
    """
    Consists of 4 resid blocks and FC layer at the end
    """

    def __init__(self, params, device="cpu"):
        super().__init__()

        in_channels = params["in_channels"]
        num_classes = params["num_classes"]
        channels = params["channels"]
        width_multiplier = params["width_multiplier"]

        channels = [int(elt * width_multiplier) for elt in channels]

        self.conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            channels[0],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(channels[0])),
                    ("act", nn.ReLU()),
                ]
            )
        )

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("dsconv1", DepthwiseSeparableConv(channels[0], channels[1], 1)),
                    ("dsconv2", DepthwiseSeparableConv(channels[1], channels[2], 1)),
                    ("dsconv3", DepthwiseSeparableConv(channels[2], channels[2], 1)),
                    ("dsconv4", DepthwiseSeparableConv(channels[2], channels[3], 1)),
                    ("dsconv5", DepthwiseSeparableConv(channels[3], channels[3], 1)),
                    ("dsconv6", DepthwiseSeparableConv(channels[3], channels[4], 1)),
                    ("dsconv7_1", DepthwiseSeparableConv(channels[4], channels[4], 1)),
                    ("dsconv7_2", DepthwiseSeparableConv(channels[4], channels[4], 1)),
                    ("dsconv7_3", DepthwiseSeparableConv(channels[4], channels[4], 1)),
                    ("dsconv7_4", DepthwiseSeparableConv(channels[4], channels[4], 1)),
                    ("dsconv7_5", DepthwiseSeparableConv(channels[4], channels[4], 1)),
                    ("dsconv8", DepthwiseSeparableConv(channels[4], channels[5], 1)),
                    ("dsconv9", DepthwiseSeparableConv(channels[5], channels[5], 1)),
                ]
            )
        )

        # Average pooling (used in classification head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification task
        self.linear = nn.Linear(in_features=channels[5], out_features=num_classes)

    def forward(self, x):
        # Perform convolutions
        x = self.conv(x)
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
