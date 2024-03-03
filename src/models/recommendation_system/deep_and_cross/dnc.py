"""
model - Deep and Cross
reference : https://arxiv.org/abs/1708.05123

This model introduces feature crossing with DNN 
"""

# torch packages
import torch
import torch.nn as nn


def get_params():
    # The following params are for training DE-EN model on Multi30K data
    params = {
        "items": 40000,
        "users": 2000000,
        "dim": 16,
        "layers": [(16, 32), (32, 64), (63, 32), (32, 16)],
    }
    return params


class DeepNet(nn.Module):
    def __init__(self, input_shape, deep_layers):
        super().__init__()
        fc_list = []
        fc_list.append(nn.Linear(input_shape, deep_layers[0]))
        fc_list.append(nn.BatchNorm1d(deep_layers[0]))
        fc_list.append(nn.ReLU())
        for i in range(1, len(deep_layers)):
            fc_list.append(nn.Linear(deep_layers[i - 1], deep_layers[i]))
            fc_list.append(nn.BatchNorm1d(deep_layers[i]))
            fc_list.append(nn.ReLU())
        self.deep = nn.Sequential(*fc_list)

    def forward(self, x):
        out = self.deep(x)
        return out


class CrossNet(nn.Module):
    """
    Cross layer part in Cross and Deep Network
    This module is x_0 * x_l^T * w_l + x_l + b_l
    for each layer l, and x_0 is the init input of this module
    """

    def __init__(self, input_shape, cross_layers):
        super().__init__()

    def forward(self):
        pass


class DeepAndCross(nn.Module):
    def __init__(
        self,
        params,
        device="cpu",
    ):
        super().__init__()
        self.items = params["items"]
        self.users = params["users"]
        self.dim = params["dim"]
        self.layers = params["layers"]

        # Modules required to build Encoder
        self.item_embedding = nn.Embedding(self.items, self.dim)
        self.user_embedding = nn.Embedding(self.users, self.dim)

        self.fc_layers = nn.ModuleList()
        for in_size, out_size in self.layers:
            self.fc_layers.append(nn.Linear(in_size, out_size))
        # Final layer
        self.fc_layers.append(nn.Linear(self.layers[-1][1], 1))

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        item_index,
        user_index,
    ):

        user_embedding = self.user_embedding(user_index)
        item_embedding = self.item_embedding(item_index)

        x = torch.cat([user_embedding, item_embedding], dim=-1)

        for layer in self.fc_layers:
            x = layer(x)
            x = nn.ReLU()(x)

        out = self.sigmoid(x)
        return out
