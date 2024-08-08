import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderGRU(nn.Module):
    def __init__(self, params, dim_in, dim_feedforward, layer_num, dim_linear_list=[]):
        super().__init__()

        # pattern extractor         
        self.pattern_extractor = nn.GRU(dim_in, dim_feedforward, layer_num, 
                                            batch_first=True, dropout=params.dropout)
        
        # hidden representation learner with linear layer
        self.linear = len(dim_linear_list) > 1
        if self.linear:
            self.hid_learner = nn.ModuleList()
            for i in range(len(dim_linear_list)-1):
                self.hid_learner.append(nn.Linear(dim_linear_list[i], dim_linear_list[i+1]))
                if i < len(dim_linear_list) - 2:
                    self.hid_learner.append(nn.Tanh())

    def forward(self, X, X_len=None):

        # learn with GRU
        if X_len != None:
            X = pack_padded_sequence(X, X_len, batch_first=True, enforce_sorted=False)     
        _, X_hid = self.pattern_extractor(X)    # h in GRU:(D∗num_layers, N, Hout)         
        X_hid = X_hid[-1, :, :]                  

        # learn with linear
        if self.linear:
            for layer in self.hid_learner: 
                X_hid = layer(X_hid)    

        return X_hid
