import torch
import torch.nn as nn
import torch.nn.functional as F

class nnEmbedder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        self.embedder = nn.Embedding(dim_in, dim_out, padding_idx=0)
       

    def forward(self, x):

        x_embed = self.embedder(x)

        return x_embed
