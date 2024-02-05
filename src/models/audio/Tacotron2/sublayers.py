"""
reference code : https://github.com/NVIDIA/tacotron2/blob/master/model.py

This model was developed for Audio task for TTS Application. 
The following code is referenced from the above refernce code. 
"""

# importing required libraries
import math

# torch packages
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        """
        Input Args:
        vocab_size : Total vocab size
        embed_size : Embedding size of token embedding
        """
        super(Embedding, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)

    def forward(self, x):
        return self.input_embeddings[x]
    
class Locationlayer(nn.Module):
    def __init__(
                self, 
                attention_filters, 
                attention_kernel_size, 
                attention_embedding):
        super(Locationlayer, self).__init__()

        padding = int(attention_kernel_size-1/2)
        self.conv = nn.Conv1d(
                                2, 
                                attention_filters, 
                                kernel_size = attention_kernel_size,
                                padding = padding)
        self.linear = nn.Linear(
                                attention_filters, 
                                attention_embedding)

    def forward(self, attention_wts):
        attention_wts = self.conv(attention_wts)
        attention_wts = attention_wts.transpose(1,2)
        attention_wts = self.linear(attention_wts)
        return attention_wts
    

class LocationSensitiveAttention(nn.Module):
    def __init__(
                self,
                attention_rnn_dim,
                attention_dim,
                embedding_dim,
                attention_filters,
                attention_kernel_size
                ):
        super(LocationSensitiveAttention, self).__init__()

        self.q = nn.Linear(
                            attention_rnn_dim,
                            attention_dim
                           )
        # self.k = nn.Linear(
        #                     embedding_dim,
        #                     attention_dim
        #                    )
        self.v = nn.Linear(
                            attention_dim,
                            1
                           )
        self.Locationlayer = Locationlayer(
                            attention_filters = attention_filters, 
                            attention_kernel_size = attention_kernel_size, 
                            attention_embedding = attention_dim)
        
        # Initialize mask for masking attention alignment
        self.score_mask_value = -float("inf")
        self.softmax = nn.Softmax(dim = 1)
        
    def get_alignment(
                self, 
                query, 
                processed_memory, 
                attention_wts):
        """
        Input Args:

            query: decoder output (batch, n_mel_channels * n_frames_per_step)
            processed_memory: processed encoder outputs (B, T_in, attention_dim)
            attention_wts: cumulative and prev. att weights (B, 2, max_time) <- Remeber input filter of location layer is 2

        Output Args:

            alignment:  (batch, max_time)
        """
        query = self.q(query.unsqueeze(1))                 # <- (B, T, dim)
        attention_wts = self.Locationlayer(attention_wts)  # <- (B, T, atten_embed)

        additive_inputs = query + attention_wts + processed_memory

        out = self.v(torch.tanh(additive_inputs))
        out = out.squeeze(-1)
        return out


    def forward(
            self, 
            attention_hidden_state, 
            memory, 
            processed_memory,
            attention_weights_cat, 
            mask):
        """
        Input Args:
        
            attention_hidden_state: attention rnn last output
            memory: encoder outputs
            processed_memory: processed encoder outputs
            attention_weights_cat: previous and cummulative attention weights
            mask: binary mask for padded data
        """
        
        alignment = self.get_alignment(
                            query = attention_hidden_state, 
                            processed_memory = processed_memory, 
                            attention_wts = attention_weights_cat)
        
        if mask is not None:
            alignment.data.masked_fill_(
                                    mask, 
                                    self.score_mask_value)
            
        # Similar to what we saw in attention to all you need paper in self_attention
        attention_weights = self.softmax(alignment)   

        # Batch matrix multiplication
        attention_context = torch.bmm(
                            attention_weights.unsqueeze(1), 
                            memory)
        
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
        
class PreNet(nn.Module):
    def __init__(
            self,
            in_dim,
            sizes,
            prenet_dropout):
        super(PreNet, self).__init__()
        input_size = [in_dim] + sizes[:-1]

        self.conv_filter_list = []
        for in_size, out_size in zip(input_size, sizes):
            conv = nn.Sequential(
                                nn.Linear(in_size, out_size),
                                nn.ReLU(inplace=True),
                                )
            self.conv_filter_list.append(conv)
        self.conv_filter_list = nn.ModuleList(self.conv_filter_list)
        self.dropout = nn.Dropout(p = prenet_dropout)

    def forward(self, x):
        for conv in self.conv_filter_list:
            x = self.dropout(conv(x))
        return x


class PostNet(nn.Module):
    def __init__(
                self, 
                num_layers, 
                postnet_dropout,
                postnet_kernel_size,
                postnet_embedding_dim,
                n_mel_channels,
                ):
        super(PostNet, self).__init__()

        padding = int(postnet_kernel_size-1/2)

        self.conv = nn.ModuleList()
        layer = nn.Sequential(
                            nn.Conv1d(
                                n_mel_channels, 
                                postnet_embedding_dim,
                                kernel_size = postnet_kernel_size,
                                padding = padding),
                            nn.BatchNorm1d(postnet_embedding_dim),
                            nn.Tanh(),
                            )
        self.conv.append(layer)

        for _ in range(1, num_layers-1):
            layer = nn.Sequential(
                                nn.Conv1d(
                                    postnet_embedding_dim, 
                                    postnet_embedding_dim,
                                    kernel_size = postnet_kernel_size,
                                    padding = padding),
                                nn.BatchNorm1d(postnet_embedding_dim),
                                nn.Tanh(),
                                )
            self.conv.append(layer)
        
        layer = nn.Sequential(
                            nn.Conv1d(
                                postnet_embedding_dim, 
                                n_mel_channels,
                                kernel_size = postnet_kernel_size,
                                padding = padding),
                            nn.BatchNorm1d(n_mel_channels),
                            )
        self.conv.append(layer)
        self.dropout = nn.Dropout(p = postnet_dropout)
            
    def forward(self, x):
        for conv in self.conv:
            x = self.dropout(conv(x))
        return x
