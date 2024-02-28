"""
model - Ripplenet
reference : https://arxiv.org/abs/1803.03467

This model was developed by combining graph based traversal with embedding based architecture
for recommendations task. 
"""

import numpy as np
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


class NeuralCollabFilter(nn.Module):
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
        
        x = torch.cat([user_embedding, item_embedding], dim = -1)

        for layer in self.fc_layers:
            x = layer(x)
            x = nn.ReLU()(x)
        
        out = self.sigmoid(x)
        return out

    