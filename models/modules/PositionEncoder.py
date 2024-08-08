import torch
import torch.nn as nn

class PositionEncoder(nn.Module):
    '''
    Revised from pytorch tutorial
    '''
    def __init__(self, dim, dropout, seq_len):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # (B, L, E) -> (L, B, E)
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        # (L, B, E) -> (B, L, E)
        x = x.transpose(0, 1)

        return self.dropout(x)