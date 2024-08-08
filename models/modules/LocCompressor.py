import torch
import torch.nn as nn
import torch.nn.functional as F


class LocCompressor(nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        # Compress Location Info
        self.compressor = nn.ModuleList()
        for i in range(len(dim_list)-1):
            self.compressor.append(nn.Linear(dim_list[i], dim_list[i+1]))
            # not the last layer, add ReLU
            if i < len(dim_list) - 2: 
                self.compressor.append(nn.ReLU())

    def forward(self, X):
        # compress location info
        for layer in self.compressor:
            X = layer(X)       # X: (B, S, Hs) -> X_con: (B, S, Ht)

        return X

