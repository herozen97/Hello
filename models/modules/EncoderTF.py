import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules



class EncoderTF(nn.Module):
    def __init__(self, params, dim_in, head_num, dim_feedforward, layer_num, seq_len, dim_linear_list=[]):
        super().__init__()
        
        # position encoder
        self.pos_encoder = modules.PositionEncoder(dim_in, params.dropout, seq_len)
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(dim_in, head_num, 
                            dim_feedforward=dim_feedforward, dropout=params.dropout, batch_first=True)
        self.pattern_extractor = nn.TransformerEncoder(encoder_layer, layer_num)
        
        # hidden representation learner
        self.linear = len(dim_linear_list) > 1
        if self.linear:
            self.hid_learner = nn.ModuleList()
            for i in range(len(dim_linear_list)-1):
                self.hid_learner.append(nn.Linear(dim_linear_list[i], dim_linear_list[i+1]))
                if i < len(dim_linear_list) - 2:
                    # self.hid_learner.append(nn.Tanh())
                    self.hid_learner.append(nn.ReLU())


    def forward(self, X, mask=None, mode='first'):

        # shape: X=(B,S,E), mask=(B, S)
        # learn with attention
        X_pe = self.pos_encoder(X)
        X_hid = self.pattern_extractor(X_pe, src_key_padding_mask=mask)   # (B, S, E)

        # plan A: mean
        if mode == 'mean':
            X_hid = torch.mean(X_hid, dim=1)    # (B, E)
        # plan B: last
        elif mode == 'last':
            X_hid = X_hid[:, -1, :]
        elif mode == 'first':
            X_hid = X_hid[:, 0, :]
        elif mode == 'all':
            X_hid = X_hid

        # learn with linear
        if self.linear:
            for layer in self.hid_learner: 
                X_hid = layer(X_hid)    # (B, E) -> (B, Hx)

        return X_hid
