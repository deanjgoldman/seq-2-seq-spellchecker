import random

import torch
import torch.nn as nn

from .encoder import EncoderGRU
from .decoder import DecoderGRU


class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim,
        n_layers=(2,2), bidirectional=False, teacher_forcing_ratio=0.3):
        super(Seq2SeqGRU, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.encoder = EncoderGRU(
            input_dim,
            embed_dim,
            self.hidden_dim,
            n_layers=n_layers[0],
            bidirectional=bidirectional)
        self.decoder = DecoderGRU(
            input_dim,
            output_dim,
            embed_dim,
            self.hidden_dim,
            n_layers=n_layers[0],
            bidirectional=bidirectional)


    def forward(self, input, target, inference=False):
        out_len = target.shape[1]
        device = torch.device('cuda:0')

        encoder_output, hidden = self.encoder(input)
        
        output_matrix = torch.zeros((target.shape[0], out_len, self.output_dim))

        output_idx = torch.zeros((target.shape[0], out_len))

        decoder_input = torch.unsqueeze(target[:, 0], dim=1)

        for di in range(1, out_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)

            topv, topi = decoder_output.topk(1)

            # set next input token
            teacher_force = (random.random() < self.teacher_forcing_ratio) & (not inference)
            if teacher_force:
                decoder_input = torch.unsqueeze(target[:, di], dim=1)
            else:
                decoder_input = torch.zeros_like(torch.unsqueeze(target[:, di], dim=1))
                decoder_input = decoder_input.scatter(dim=2, index=topi, src=torch.ones((target.shape[0], 1, 1)).to(device))
            
            output = decoder_output.squeeze().to(device)

            output_matrix[:, di, :] = output

            output_idx[:, di] = topi.squeeze()

        return output_idx.to(device), output_matrix.to(device)