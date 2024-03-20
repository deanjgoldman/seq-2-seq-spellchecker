import random

import torch
import torch.nn as nn

from .encoder import EncoderLSTM
from .decoder import DecoderLSTM


class Seq2SeqCNN(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=5,
        padding=2,
        n_layers=30):
        super(Seq2SeqCNN, self).__init__()

        layers = []
        for _ in range(n_layers):
            conv = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding)
            bn = torch.nn.BatchNorm1d(in_channels, out_channels)
            layers.append(conv)
            layers.append(bn)

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x