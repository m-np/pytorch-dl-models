import math
import copy
import time
import random
import spacy
import numpy as np
import os 

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from src.models.nlp.Bert.attention_sublayers import (
                                                    MultiHeadAttention,
                                                    BertEmbedding,
                                                    PositionwiseFeedForward,
                                                    )


class EncoderLayer(nn.Module):
    """
    This building block in the encoder layer consists of the following
    1. MultiHead Attention
    2. Sublayer Logic
    3. Positional FeedForward Network
    """
    def __init__(self, dk, dv, h, dim_multiplier = 4, pdropout=0.1):
        """
        Above notations are as per the original attention network paper
        https://arxiv.org/abs/1706.03762

        Input Args:
            dk: Key dimension
            dv: value dimension
            h: number of heads 
            dim_multiplier: internal linear layer multiplier in FF Network
            pdropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(dk, dv, h, pdropout)
        # Reference page 5 chapter 3.2.2 Multi-head attention
        dmodel = dk*h
        # Reference page 5 chapter 3.3 positionwise FeedForward
        dff = dmodel * dim_multiplier
        self.attn_norm = nn.LayerNorm(dmodel)
        self.ff = PositionwiseFeedForward(dmodel, dff, pdropout=pdropout)
        self.ff_norm = nn.LayerNorm(dmodel)
        self.dmodel = dmodel
        
        self.dropout = nn.Dropout(p = pdropout)
        
    def forward(self, src_inputs, src_mask=None):
        """
        Forward pass as per page 3 chapter 3.1
        """
        mha_out, attention_wts = self.attention(
                                query = src_inputs, 
                                key = src_inputs, 
                                val = src_inputs, 
                                mask = src_mask)
        
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        # Actual paper design is the following
        intermediate_out = self.attn_norm(src_inputs + self.dropout(mha_out))
        
        pff_out = self.ff(intermediate_out)
        
        # Perform Add Norm again
        out = self.ff_norm(intermediate_out + self.dropout(pff_out))
        return out, attention_wts


class Encoder(nn.Module):
    def __init__(self, dk, dv, h, num_encoders, dim_multiplier = 4, pdropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dk, 
                         dv, 
                         h, 
                         dim_multiplier, 
                         pdropout) for _ in range(num_encoders)
        ])
        
    def forward(self, src_inputs, src_mask = None):
        """
        Input from the Embedding layer
        src_inputs = (B - batch size, S/T - max token sequence length, D- model dimension)
        """
        src_representation = src_inputs
        
        # Forward pass through encoder stack
        for enc in self.encoder_layers:
            src_representation, attn_probs = enc(src_representation, src_mask)
            
        self.attn_probs = attn_probs
        return src_representation


class BERT(nn.Module):
    def __init__(
                self,
                dk, 
                dv, 
                h,
                vocab_size,
                seq_len,
                num_encoders,
                dim_multiplier = 4, 
                pdropout = 0.1
                ):
        super().__init__()

        # Reference page 5 chapter 3.2.2 
        # Multi-head attention from original transformer paper
        dmodel = dk*h
        self.dmodel = dmodel

        self.embedding = BertEmbedding(
                                vocab_size, 
                                embed_size = self.dmodel, 
                                seq_len = seq_len, 
                                pdropout=pdropout)

        self.encoder = Encoder(
                                dk, 
                                dv, 
                                h, 
                                num_encoders, 
                                dim_multiplier=dim_multiplier, 
                                pdropout=pdropout)

    def forward(self, token_ids_batch, segment_label):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (token_ids_batch > 0).unsqueeze(1).repeat(
                                            1, 
                                            token_ids_batch.size(1), 
                                            1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        token_representations = self.embedding(
                                    token_ids_batch, 
                                    segment_label)

        # Encode 
        encoded_src = self.encoder(token_representations, mask)
        return encoded_src


class MaskedLanguageModel(nn.Module):
    """
    Predicting the token from masked input sequence

    Its a multiclass classification where classes = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        """
        Input Args:
        hidden: Ouptut size of BERT model
        vocab_size: Output classes for predictions 
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class NextSentencePrediction(nn.Module):
    """
    Binary class classification to predict if the sentence is next sentence
        : is_next, not_next
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim = -1)
        
    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class BERTLM(nn.Module):
    """
    BERT Language Model
    The idea is to optimize both tasks
    1. Next Sentence Prediction
    2. Masked Language Modeling
    """
    def __init__(
                self,
                dk, 
                dv, 
                h,
                vocab_size,
                seq_len,
                num_encoders,
                dim_multiplier = 4, 
                pdropout = 0.1
                ):
        super().__init__()
        self.bert = BERT(
                        dk, 
                        dv, 
                        h,
                        vocab_size,
                        seq_len,
                        num_encoders,
                        dim_multiplier = dim_multiplier, 
                        pdropout = pdropout)
        
        self.mlm = MaskedLanguageModel(
                        hidden = self.bert.dmodel, 
                        vocab_size = vocab_size
                        )

        self.nsp = NextSentencePrediction(
                        hidden = self.bert.dmodel
                        )

    def forward(
                self, 
                token_ids_batch,
                segment_label):
        x = self.bert(token_ids_batch, segment_label)
        return (self.mlm(x), self.nsp(x))