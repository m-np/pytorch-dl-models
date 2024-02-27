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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def get_params():
    # The following params are for training DE-EN model on Multi30K data
    params = {
        "n_entity": 5400,
        "n_relations": 9,
        "dim": 16,
        "n_hop": 2,
        "n_memory": 32,
        "using_all_hop": True,
        "kge_weight": 0.01,
        "l2_weight": 1e-7,
    }
    return params


class RippleNet(nn.Module):
    def __init__(
        self,
        params,
        device="cpu",
    ):
        super().__init__()
        self.n_entity = params["n_entity"]
        self.n_relations = params["n_relations"]
        self.dim = params["dim"]
        self.n_hop = params["n_hop"]
        self.n_memory = params["n_memory"]
        self.using_all_hop = params["using_all_hop"]
        self.kge_weight = params["kge_weight"]
        self.l2_weight = params["l2_weight"]

        # Modules required to build Encoder
        self.item_embedding = nn.Embedding(self.n_entity, self.dim)
        self.relation_embedding = nn.Embedding(self.n_relations, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.BCELoss()

    def forward(
        self,
        items,
        labels,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):

        self.item_embed = self.item_embedding(items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []

        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(self.item_embedding(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(
                self.relation_embedding(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            self.t_emb_list.append(self.item_embedding(memories_t[i]))

        o_list = self._key_addressing()
        scores = self.predict(self.item_embed, o_list)
        scores = torch.sigmoid(scores)
        out = self._build_loss(scores=scores, labels=labels)
        out["scores"] = scores
        return out

    def _key_addressing(
        self,
    ):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = torch.unsqueeze(self.item_embed, axis=2)

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, axis=2)

            # [batch_size, dim]
            o = (self.t_emb_list[hop] * probs_expanded).sum(dim=1)

            self.item_embed = self._update_item_embedding(self.item_embed, o)
            o_list.append(o)
        return o_list

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = (item_embeddings * y).sum(axis=1)
        return scores

    def _build_loss(self, scores, labels):
        self.base_loss = self.criterion(scores, labels)

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = torch.unsqueeze(self.h_emb_list[hop], axis=2)
            t_expanded = torch.unsqueeze(self.t_emb_list[hop], axis=3)
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, self.r_emb_list[hop]), t_expanded)
            )
            self.kge_loss += torch.sigmoid(hRt).mean()
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += (self.h_emb_list[hop] * self.h_emb_list[hop]).sum()
            self.l2_loss += (self.t_emb_list[hop] * self.t_emb_list[hop]).sum()
            self.l2_loss += (self.r_emb_list[hop] * self.r_emb_list[hop]).sum()
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss
        return dict(
            base_loss=self.base_loss,
            kge_loss=self.kge_loss,
            l2_loss=self.l2_loss,
            loss=self.loss,
        )

    def evaluate(
        self,
        items,
        labels,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        out = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = out["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc
