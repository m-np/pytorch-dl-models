"""
model - ResNet50
reference : https://medium.com/@freshtechyy/a-detailed-introduction-to-resnet-and-its-implementation-in-pytorch-744b13c8074a

This model was developed for image classification task on ImageNet Dataset
"""

# torch packages
import torch.nn as nn


def get_params():
    params = {
        "in_channels": 3,
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


class ResidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, skip_block=False):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Don't add relu to this block since we will be adding skip_block
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels * expansion),
        )

        self.skip_layer = None
        if skip_block:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(out_channels * expansion),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block output
        """
        x_2 = x.clone()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.skip_layer:
            x_2 = self.skip_layer(x_2)

        x += x_2
        return self.relu(x)


class ResNet50(nn.Module):
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
                in_channels=in_channels,
                out_channels=layer_filter,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(layer_filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.resid_conv1 = self.create_resid_layers(
            in_channels=layer_filter,
            out_channels=conv1_filters,
            stride=1,
            expansion=expansion,
            num_blocks=num_blocks_1,
        )

        self.resid_conv2 = self.create_resid_layers(
            in_channels=conv1_filters * expansion,
            out_channels=conv2_filters,
            stride=2,
            expansion=expansion,
            num_blocks=num_blocks_2,
        )

        self.resid_conv3 = self.create_resid_layers(
            in_channels=conv2_filters * expansion,
            out_channels=conv3_filters,
            stride=2,
            expansion=expansion,
            num_blocks=num_blocks_3,
        )

        self.resid_conv4 = self.create_resid_layers(
            in_channels=conv3_filters * expansion,
            out_channels=conv4_filters,
            stride=2,
            expansion=expansion,
            num_blocks=num_blocks_4,
        )

        # Average pooling (used in classification head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification task
        # in_channel = 4, By default, ResBlock.expansion = 4 for ResNet-50, 101, 152,
        # Check line 77
        self.linear = nn.Linear(
            in_features=conv4_filters * expansion, out_features=num_classes
        )

    def create_resid_layers(
        self, in_channels, out_channels, stride, expansion, num_blocks
    ):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                # Downsample the first layer by adding skip block
                resid_layer = ResidBlock(
                    in_channels,
                    out_channels,
                    stride=stride,
                    expansion=expansion,
                    skip_block=True,
                )
            else:
                resid_layer = ResidBlock(
                    out_channels * expansion,
                    out_channels,
                    stride=1,
                    expansion=expansion,
                    skip_block=False,
                )
            layers.append(resid_layer)
        return nn.Sequential(*layers)

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
