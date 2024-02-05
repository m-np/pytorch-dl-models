"""
model - Tacotron2
reference : https://arxiv.org/abs/1712.05884

reference code : https://github.com/NVIDIA/tacotron2/blob/master/model.py

This model was developed for Audio task for TTS Application. The following code is referenced from the above refernce
code. 
"""

import math

# torch packages
import torch
import torch.nn as nn

from torch.autograd import Variable

from src.models.audio.Tacotron2.sublayers import (Embedding, 
                                                  LocationSensitiveAttention, 
                                                  PreNet,
                                                  PostNet)


def get_params():
    params = {
        "n_symbols": 1000,
        "n_mel_channels": 80,
        "symbols_embedding_dim": 512,

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 512,
        "p_encoder_dropout": 0.5,

        # Decoder parameters
        "n_frames_per_step": 1,  # currently only 1 is supported
        "decoder_rnn_dim": 1024,
        "prenet_dim": 256,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,
        "prenet_dropout": 0.5,

        # Attention parameters
        "attention_rnn_dim": 1024,
        "attention_dim": 128,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,
        "postnet_dropout": 0.5,

    }

    return params

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

class Encoder(nn.Module):
    def __init__(
            self, 
            num_convs,
            conv_embed_dim, 
            conv_kernel_size,
            lstm_embed_dim,
            pdropout = 0.5):
        super(Encoder, self).__init__()

        padding = int(conv_embed_dim-1/2)

        self.conv_filter_list = []
        for _ in range(num_convs):
            conv_filter =   nn.Sequential(
                                nn.Conv1d(
                                    conv_embed_dim, 
                                    conv_embed_dim,
                                    kernel_size = conv_kernel_size,
                                    padding = padding),
                                nn.BatchNorm1d(conv_embed_dim),
                                nn.ReLU(inplace=True),
                                )
            self.conv_filter_list.append(conv_filter)
        self.conv_filter_list = nn.ModuleList(self.conv_filter_list)
        self.dropout = nn.Dropout(p = pdropout)

        self.lstm = nn.LSTM(
                        lstm_embed_dim,
                        int(lstm_embed_dim / 2), 
                        1,
                        batch_first=True, 
                        bidirectional=True)
        
    def forward(self, x, input_lengths):
        for conv in self.conv_filter_list:
            x = self.dropout(conv(x))
        
        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
                            x, 
                            input_lengths, 
                            batch_first=True)
        
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)

        out, _ = nn.utils.rnn.pad_packed_sequence(
                                out, 
                                batch_first=True)

        return out

    def inference(self, x):
        for conv in self.conv_filter_list:
            x = self.dropout(conv(x))
        
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)

        return out



class Decoder(nn.Module):
    def __init__(
                self,
                n_mel_channels,
                n_frames_per_step,
                attention_rnn_dim,
                attention_dim,
                encoder_embedding_dim,
                attention_location_n_filters,
                attention_location_kernel_size,
                prenet_dim,
                decoder_rnn_dim,
                prenet_dropout,
                max_decoder_steps,
                gate_threshold,
                p_attention_dropout,
                p_decoder_dropout,
                ):
        super(Decoder, self).__init__()

        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_dim = attention_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_location_n_filters = attention_location_n_filters
        self.attention_location_kernel_size = attention_location_kernel_size
        self.prenet_dim = prenet_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dropout = prenet_dropout
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.attention = LocationSensitiveAttention(
                                    attention_rnn_dim = attention_rnn_dim, 
                                    attention_dim = attention_dim, 
                                    embedding_dim = encoder_embedding_dim, 
                                    attention_filters = attention_location_n_filters, 
                                    attention_kernel_size = attention_location_kernel_size)

        self.attention_lstm = nn.LSTM(
                                    prenet_dim + encoder_embedding_dim,
                                    attention_rnn_dim, 
                                    1)
        
        self.decoder_lstm = nn.LSTM(
                                    attention_rnn_dim + encoder_embedding_dim,
                                    decoder_rnn_dim, 
                                    1)
        
        self.prenet = PreNet(
                            in_dim = n_mel_channels * n_frames_per_step, 
                            sizes = [prenet_dim, prenet_dim], 
                            prenet_dropout = prenet_dropout
                            )
        
        self.linear_projection = nn.Linear(
                        decoder_rnn_dim + encoder_embedding_dim,
                        n_mel_channels * n_frames_per_step)
        # For stop token
        self.gate_projection = nn.Linear(  
                                        decoder_rnn_dim + encoder_embedding_dim, 
                                        1, bias=True)
        
    def get_go_frame(self, memory):
        """ 
        Gets all zeros frames to use as first decoder input

        Input Args:
            memory: decoder outputs

        RETURNS
            decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input
    
    def initialize_decoder_states(self, memory, mask):
        """ 
        Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        
        Input Args:
        
            memory: Encoder outputs
            mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ 
        Prepares decoder inputs, i.e. mel outputs

        Input Args:
            decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        
            inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ 
        Prepares decoder outputs for output

        Input Args:
        
            mel_outputs:
            gate_outputs: gate output energies
            alignments:

        RETURNS
        
            mel_outputs:
            gate_outpust: gate output energies
            alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
                            mel_outputs.size(0), 
                            -1, 
                            self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments
    
    def decode(self, decoder_input):
        """ 
        Decoder step using stored states, attention and memory

        Input Args:
        
            decoder_input: previous mel output

        RETURNS
        
            mel_output:
            gate_output: gate output energies
            attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weights

    def forward(
                self, 
                memory, 
                decoder_inputs, 
                memory_lengths):
        """ 
        Decoder forward pass for training

        Input Args:

            memory: Encoder outputs
            decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
            memory_lengths: Encoder output lengths for attention masking.

        RETURNS

            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        
        mel_outputs, gate_outputs, alignments = [], [], []
        
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            
            mel_output, gate_output, attention_weights = self.decode(
                                                                decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """
        Decoder inference

        Input Args:

            memory: Encoder outputs
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)

            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
    

class Tacotron2(nn.Module):
    """
    Wavenet Model
    The idea is to optimize
    1. Speech Generation
    """

    def __init__(
        self,
        params,
        device="cpu",
    ):
        super(Tacotron2, self).__init__()
        
        n_symbols = params["n_symbols"]
        n_mel_channels = params["n_mel_channels"]
        symbols_embedding_dim = params["symbols_embedding_dim"]

        # Encoder parameters
        encoder_kernel_size = params["encoder_kernel_size"]
        encoder_n_convolutions = params["encoder_n_convolutions"]
        encoder_embedding_dim = params["encoder_embedding_dim"]
        p_encoder_dropout = params["p_encoder_dropout"]

        # Decoder parameters
        n_frames_per_step = params["n_frames_per_step"]
        decoder_rnn_dim = params["decoder_rnn_dim"]
        prenet_dim = params["prenet_dim"]
        max_decoder_steps = params["max_decoder_steps"]
        gate_threshold = params["gate_threshold"]
        p_attention_dropout = params["p_attention_dropout"]
        p_decoder_dropout = params["p_decoder_dropout"]
        prenet_dropout = params["prenet_dropout"]

        # Attention parameters
        attention_rnn_dim = params["attention_rnn_dim"]
        attention_dim = params["attention_dim"]

        # Location Layer parameters
        attention_location_n_filters = params["attention_location_n_filters"]
        attention_location_kernel_size = params["attention_location_kernel_size"]

        # Mel-post processing network parameters
        postnet_embedding_dim = params["postnet_embedding_dim"]
        postnet_kernel_size = params["postnet_kernel_size"]
        postnet_n_convolutions = params["postnet_n_convolutions"]
        postnet_dropout = params["postnet_dropout"]

        self.embedding = Embedding(n_symbols, symbols_embedding_dim)
        self.Encoder = Encoder(num_convs = encoder_n_convolutions, 
                               conv_embed_dim = encoder_embedding_dim, 
                               conv_kernel_size = encoder_kernel_size, 
                               lstm_embed_dim = encoder_embedding_dim,
                               pdropout=p_encoder_dropout)
        self.Decoder = Decoder(
                                n_mel_channels = n_mel_channels, 
                                n_frames_per_step = n_frames_per_step, 
                                attention_rnn_dim = attention_rnn_dim, 
                                attention_dim = attention_dim, 
                                encoder_embedding_dim = encoder_embedding_dim, 
                                attention_location_n_filters = attention_location_n_filters, 
                                attention_location_kernel_size = attention_location_kernel_size, 
                                prenet_dim = prenet_dim, 
                                decoder_rnn_dim = decoder_rnn_dim, 
                                prenet_dropout = prenet_dropout,
                                max_decoder_steps = max_decoder_steps,
                                gate_threshold = gate_threshold,
                                p_attention_dropout = p_attention_dropout,
                                p_decoder_dropout = p_decoder_dropout,
                                )
        
        self.PostNet = PostNet(num_layers = postnet_n_convolutions, 
                               postnet_dropout = postnet_dropout, 
                               postnet_kernel_size = postnet_kernel_size, 
                               postnet_embedding_dim = postnet_embedding_dim, 
                               n_mel_channels = n_mel_channels)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs


    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs

        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.Encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.Decoder(
                                                            encoder_outputs, 
                                                            mels, 
                                                            memory_lengths=text_lengths)

        mel_outputs_postnet = self.PostNet(mel_outputs)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        
        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.Encoder.inference(embedded_inputs)
        
        mel_outputs, gate_outputs, alignments = self.Decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.PostNet(mel_outputs)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs