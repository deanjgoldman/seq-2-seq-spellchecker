import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim,
        hidden_dim, n_layers, bidirectional=False):
        super(DecoderLSTM, self).__init__()
        
        self.output_dim = output_dim
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
        if bidirectional:
            hidden_dim *= 2
        self.fc_out = nn.Linear(hidden_dim, output_dim)

            
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc_out(output)
        
        return prediction, hidden



class DecoderGRU(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim,
        hidden_dim, n_layers, bidirectional=False):
        super(DecoderGRU, self).__init__()
        
        self.output_dim = output_dim
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
        if bidirectional:
            hidden_dim *= 2
        self.fc_out = nn.Linear(hidden_dim, output_dim)

            
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output)
        
        return prediction, hidden