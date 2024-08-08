import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules


class TemporalDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # params
        self.traj_len = params.traj_len
        # decoder
        dim_in = params.dim_com_list[-1] + params.dim_embed_tim
        dim_linear_list = [dim_in] + params.dim_TD_linear
        self.tim_decoder = modules.EncoderTF(params, dim_in, head_num=params.head_num_tim, 
                                             dim_feedforward=params.dim_dec_tim,
                                             layer_num=params.num_dec_tim, seq_len=params.traj_len, 
                                             dim_linear_list=dim_linear_list)

    def forward(self, zt, loc_info, loc_chain):

        # get mask from loc_chain
        padding_mask = (loc_chain == 0)     # (B, S)

        # concat time and predicted loc info
        X_hid = torch.cat([zt.unsqueeze(1).expand(-1, loc_info.shape[1], -1), loc_info], dim=-1)   # (B, S=24, Ht+Hs)
        # X_hid = zt.unsqueeze(1).expand(-1, loc_info.shape[1], -1)
        # decode duration chain in the next day
        tim_chain = self.tim_decoder(X_hid, padding_mask, mode='all') * self.traj_len  # tim_chain:(B,S,1)
        # for i in range(tim_chain.size()[0]):
        #     print(tim_chain[i])
        tim_chain = tim_chain.squeeze(-1)

        return tim_chain
        