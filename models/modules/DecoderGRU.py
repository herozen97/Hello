import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class DecoderGRU(nn.Module):
    def __init__(self, params, dim_in, dim_feedforward, layer_num, mode, dim_linear_list=[]):
        super().__init__()

        # pattern decoder         
        self.pattern_decoder = nn.GRU(dim_in, dim_feedforward, layer_num, 
                                            batch_first=True, dropout=params.dropout)
        
        # hidden decoder
        self.hid_decoder = nn.ModuleList()
        for i in range(len(dim_linear_list)-1):
            self.hid_decoder.append(nn.Linear(dim_linear_list[i], dim_linear_list[i+1]))
            if i < len(dim_linear_list) - 2:
                self.hid_decoder.append(nn.Tanh())

        if mode == 'temporal':
            # end with sigmoid
            self.hid_decoder.append(nn.Sigmoid())

    def forward(self, X):

        # decode with GRU
        X_hid, _ = self.pattern_decoder(X)    # X_hid:(B, S, H)           

        # decode with linear
        for layer in self.hid_decoder:
            X_hid = layer(X_hid)    

        return X_hid
