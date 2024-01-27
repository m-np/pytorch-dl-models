"""
model - Inception V1 
reference : https://arxiv.org/abs/1409.4842

This model was developed for image classification task on ImageNet Dataset
"""

# torch packages
import torch
import torch.nn as nn


def get_params():
    params = {
        "in_channels": 3,
        "num_classes": 1000,
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
        self, in_channels, out_channels, kernel_size, stride, padding, bias=False
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
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
            in_channels=in_channels,
            out_channels=out1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # branch2 : k=1,s=1,p=0 -> k=3,s=1,p=1
        self.branch2 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out21,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBlock(
                in_channels=out21,
                out_channels=out22,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        # branch3 : k=1,s=1,p=0 -> k=5,s=1,p=2
        self.branch3 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out31,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBlock(
                in_channels=out31,
                out_channels=out32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )

        # branch4 : pool(k=3,s=1,p=1) -> k=1,s=1,p=0
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(
                in_channels=in_channels,
                out_channels=out4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        print(out1.shape)
        print(out2.shape)
        print(out3.shape)
        print(out4.shape)
        return torch.cat([out1, out2, out3, out4], dim=1)


class Inception_v1(nn.Module):
    """
    Consists of 4 resid blocks and FC layer at the end
    """

    def __init__(self, params, device="cpu"):
        super().__init__()

        in_channels = params["in_channels"]
        num_classes = params["num_classes"]

        self.conv1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(
                in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Start Adding Inception block
        self.inception3a = InceptionBlock(
            in_channels=192, out1=64, out21=96, out22=128, out31=16, out32=32, out4=32
        )
        self.inception3b = nn.Sequential(
            InceptionBlock(
                in_channels=256,
                out1=128,
                out21=128,
                out22=192,
                out31=32,
                out32=96,
                out4=64,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception4a = InceptionBlock(
            in_channels=480, out1=192, out21=96, out22=208, out31=16, out32=48, out4=64
        )
        self.inception4b = InceptionBlock(
            in_channels=512, out1=160, out21=112, out22=224, out31=24, out32=64, out4=64
        )
        self.inception4c = InceptionBlock(
            in_channels=512, out1=128, out21=128, out22=256, out31=24, out32=64, out4=64
        )
        self.inception4d = InceptionBlock(
            in_channels=512, out1=112, out21=144, out22=288, out31=32, out32=64, out4=64
        )
        self.inception4e = nn.Sequential(
            InceptionBlock(
                in_channels=528,
                out1=256,
                out21=160,
                out22=320,
                out31=32,
                out32=128,
                out4=128,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception5a = InceptionBlock(
            in_channels=832,
            out1=256,
            out21=160,
            out22=320,
            out31=32,
            out32=128,
            out4=128,
        )
        self.inception5b = nn.Sequential(
            InceptionBlock(
                in_channels=832,
                out1=384,
                out21=192,
                out22=384,
                out31=48,
                out32=128,
                out4=128,
            ),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )

        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x
