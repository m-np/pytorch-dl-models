"""
model - Wide and Deep
reference : https://arxiv.org/abs/1606.07792

This model uses feature combinations through low-dimensional 
dense embeddings learned for the sparse features
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
        "layers": [1024, 512, 256],
        "continuous_feature_shape": 10,
    }
    return params


class WideAndDeep(nn.Module):
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
        self.continuous_feature_shape = params["continuous_feature_shape"]

        # Modules required to build Encoder
        self.item_embedding = nn.Embedding(self.items, self.dim)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.dim + self.continuous_feature_shape, self.layers[0]),
            nn.ReLU(),
            nn.Linear(self.layers[0], self.layers[1]),
            nn.ReLU(),
            nn.Linear(self.layers[1], self.layers[2]),
            nn.ReLU(),
        )

        self.out = (nn.Linear(self.items + self.layers[2], self.items),)

    def forward(
        self,
        item_index,
        continious,
        binary,
    ):

        binary_embed = self.item_embedding(item_index)
        binary_embed_mean = torch.mean(binary_embed, dim=1)
        # get logits for "deep" part: continious features + binary embeddings
        deep_logits = self.fc_layer(torch.cat((continious, binary_embed_mean), dim=1))
        # get final softmax logits for "deep" part and raw binary features
        out = self.head(torch.cat((deep_logits, binary), dim=1))
        return out
