# importing required libraries
import math

# torch packages
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    We can refer to the following blog to understand in depth about the transformer and MHA
    https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a

    Here we are clubbing all the linear layers together and duplicating the inputs and
    then performing matrix multiplications
    """

    def __init__(self, dk, dv, h, pdropout=0.1):
        """
        Input Args:

        dk(int): Key dimensions used for generating Key weight matrix
        dv(int): Val dimensions used for generating val weight matrix
        h(int) : Number of heads in MHA
        """
        super().__init__()
        assert dk == dv
        self.dk = dk
        self.dv = dv
        self.h = h
        self.dmodel = self.dk * self.h  # model dimension

        # Add the params in modulelist as the params in the conv list needs to be tracked
        # wq, wk, wv -> multiple linear weights for the number of heads
        self.WQ = nn.Linear(self.dmodel, self.dmodel)  # shape -> (dmodel, dmodel)
        self.WK = nn.Linear(self.dmodel, self.dmodel)  # shape -> (dmodel, dmodel)
        self.WV = nn.Linear(self.dmodel, self.dmodel)  # shape -> (dmodel, dmodel)
        # Output Weights
        self.WO = nn.Linear(self.h * self.dv, self.dmodel)  # shape -> (dmodel, dmodel)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=pdropout)

    def forward(self, query, key, val, mask=None):
        """
        Forward pass for MHA

        X has a size of (batch_size, seq_length, d_model)
        Wq, Wk, and Wv have a size of (d_model, d_model)

        Perform Scaled Dot Product Attention on multi head attention.

        Notation: B - batch size, S/T - max src/trg token-sequence length
        query shape = (B, S, dmodel)
        key shape = (B, S, dmodel)
        val shape = (B, S, dmodel)
        """
        # Weight the queries
        Q = self.WQ(query)  # shape -> (B, S, dmodel)
        K = self.WK(key)  # shape -> (B, S, dmodel)
        V = self.WV(val)  # shape -> (B, S, dmodel)

        # Separate last dimension to number of head and dk
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.h, self.dk)  # shape -> (B, S, h, dk)
        K = K.view(batch_size, -1, self.h, self.dk)  # shape -> (B, S, h, dk)
        V = V.view(batch_size, -1, self.h, self.dk)  # shape -> (B, S, h, dk)

        # each sequence is split across n_heads, with each head receiving seq_length tokens
        # with d_key elements in each token instead of d_model.
        Q = Q.permute(0, 2, 1, 3)  # shape -> (B, h, S, dk)
        K = K.permute(0, 2, 1, 3)  # shape -> (B, h, S, dk)
        V = V.permute(0, 2, 1, 3)  # shape -> (B, h, S, dk)

        # dot product of Q and K
        scaled_dot_product = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dk)

        # fill those positions of product as (-1e10) where mask positions are 0
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e10)

        attn_probs = self.softmax(scaled_dot_product)

        # Create head
        head = torch.matmul(
            self.dropout(attn_probs), V
        )  # shape -> (B, h, S, S) * (B, h, S, dk) = (B, h, S, dk)
        # Prepare the head to pass it through output linear layer
        head = head.permute(0, 2, 1, 3).contiguous()  # shape -> (B, S, h, dk)
        # Concatenate the head together
        head = head.view(
            batch_size, -1, self.h * self.dk
        )  # shape -> (B, S, (h*dk = dmodel))
        # Pass through output layer
        token_representation = self.WO(head)
        return token_representation, attn_probs


class Embedding(nn.Module):
    """
    Embedding lookup table which is used by the positional
    embedding block.
    Embedding lookup table is shared across input and output
    """

    def __init__(self, vocab_size, dmodel):
        """
        Embedding lookup needs a vocab size and model
        dimension size matrix for creating lookups
        """
        super().__init__()
        self.embedding_lookup = nn.Embedding(vocab_size, dmodel)
        self.vocab_size = vocab_size
        self.dmodel = dmodel

    def forward(self, token_ids):
        """
        For a given token lookup the embedding vector

        As per the paper, we also multiply the embedding vector with sqrt of dmodel
        """
        assert (
            token_ids.ndim == 2
        ), f"Expected: (batch size, max token sequence length), got {token_ids.shape}"

        embedding_vector = self.embedding_lookup(token_ids)

        return embedding_vector * math.sqrt(self.dmodel)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dmodel,
        max_seq_length=5000,
        pdropout=0.1,
    ):
        """
        dmodel(int): model dimensions
        max_seq_length(int): Maximum input sequence length
        pdropout(float): Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=pdropout)

        # Calculate frequencies
        position_ids = torch.arange(0, max_seq_length).unsqueeze(1)
        # -ve sign is added because the exponents are inverted when you multiply position and frequencies
        frequencies = torch.pow(
            10000, -torch.arange(0, dmodel, 2, dtype=torch.float) / dmodel
        )

        # Create positional encoding table
        positional_encoding_table = torch.zeros(max_seq_length, dmodel)
        # Fill the table with even entries with sin and odd entries with cosine
        positional_encoding_table[:, 0::2] = torch.sin(position_ids * frequencies)
        positional_encoding_table[:, 1::2] = torch.cos(position_ids * frequencies)

        # Registering the position enconding in state_dict but the its not included
        # in named parameter as it is not trainable
        self.register_buffer("positional_encoding_table", positional_encoding_table)

    def forward(self, embeddings_batch):
        """
        embeddings_batch shape = (batch size, seq_length, dmodel)
        positional_encoding_table shape = (max_seq_length, dmodel)
        """
        assert (
            embeddings_batch.ndim == 3
        ), f"Embeddings batch should have dimension of 3 but got {embeddings_batch.ndim}"
        assert (
            embeddings_batch.size()[-1] == self.positional_encoding_table.size()[-1]
        ), f"Embedding batch shape and positional_encoding_table shape should match, expected Embedding batch shape : {embeddings_batch.shape[-1]} while positional_encoding_table shape : {self.positional_encoding_table[-1]}"

        # Get encodings for the given input sequence length
        pos_encodings = self.positional_encoding_table[
            : embeddings_batch.shape[1]
        ]  # Choose only seq_length out of max_seq_length

        # Final output
        out = embeddings_batch + pos_encodings
        out = self.dropout(out)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dmodel, dff, pdropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=pdropout)

        self.W1 = nn.Linear(dmodel, dff)  # Intermediate layer
        self.W2 = nn.Linear(dff, dmodel)  # Output layer

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Perform Feedforward calculation

        x shape = (B - batch size, S/T - max token sequence length, D- model dimension).
        """
        out = self.W2(self.relu(self.dropout(self.W1(x))))
        return out
