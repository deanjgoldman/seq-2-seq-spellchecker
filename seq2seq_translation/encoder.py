import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim,
        n_layers, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            batch_first=True)


    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)

        return output, hidden


class EncoderGRU(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim,
        n_layers, bidirectional=False):
        super(EncoderGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            n_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            batch_first=True)


    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)

        return output, hidden