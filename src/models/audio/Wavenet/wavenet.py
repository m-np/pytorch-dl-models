"""
model - Wavenet
reference : https://arxiv.org/abs/1609.03499

This model was developed for Audio task as a generative network. However, it can
perform following task:
1. Speech Recognition
2. TTS vocoder
3. Speech Generation
"""

# torch packages
import torch.nn as nn
import math

from src.models.audio.Wavenet.sublayers import WaveBlock


def get_params():
    params = {
        "num_layers": 20,
        "stacks": 2,
        "in_channels": 512,
        "residual_channels": 512,
        "gate_channels": 512,
        "skip_channels": 512,
        "kernel_size": 3,
        "pdropout": 0.1,
        "local_conditioning_channels": -1,
        "global_conditioning_channels": -1,
        "use_speaker_embedding": False,
        "n_speakers": None,
    }
    return params


class WaveNet(nn.Module):
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
        super(WaveNet, self).__init__()

        num_layers = params["num_layers"]
        stacks = params["stacks"]
        in_channels = params["in_channels"]
        residual_channels = params["residual_channels"]
        gate_channels = params["gate_channels"]
        skip_channels = params["skip_channels"]
        kernel_size = params["kernel_size"]
        pdropout = params["pdropout"]
        local_conditioning_channels = params["local_conditioning_channels"]
        global_conditioning_channels = params["global_conditioning_channels"]
        use_speaker_embedding = params["use_speaker_embedding"]
        n_speakers = params["n_speakers"]

        layers_per_stack = num_layers // stacks

        self.first_conv = nn.Conv1d(
                                in_channels, 
                                residual_channels,
                                kernel_size=1)

        # Add resid block
        self.waveblocks = nn.ModuleList()
        for layer in range(num_layers):
            dilation = 2**(layer % layers_per_stack)
            block = WaveBlock(
                        in_channels = residual_channels, 
                        gate_channels = gate_channels, 
                        kernel_size = kernel_size, 
                        stride = 1, 
                        dilation = dilation, 
                        skip_channels = skip_channels, 
                        local_conditioning_channels = local_conditioning_channels, 
                        global_conditioning_channels = global_conditioning_channels, 
                        pdropout = pdropout, 
                        bias = True)
            self.waveblocks.append(block)

        # Add Output blocks
        self.final_block = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Conv1d(
                            skip_channels, 
                            skip_channels, 
                            kernel_size=1),        # Use skip connections as input and not x
                        nn.ReLU(inplace=True),
                        nn.Conv1d(
                            skip_channels, 
                            in_channels, 
                            kernel_size=1),
                        )

        if global_conditioning_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = nn.Embedding(
                                    n_speakers, 
                                    global_conditioning_channels, 
                                    padding_idx=None, 
                                    std=0.1)
        else:
            self.embed_speakers = None

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, c=None, g=None, softmax=False):
        """
        Args: 
            x (Tensor): One-hot encoded audio signal, shape (B x C x l)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x l)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.
        """
        # Get Batch size
        B = x.size()[0]
        
        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        
        # Feed data to network
        x = self.first_conv(x)

        skips = 0
        for f in self.conv_layers:
            x, s = f(x, c, g)
            skips += s
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        x = self.final_block(x)

        x = self.softmax(x) if softmax else x

        return x

