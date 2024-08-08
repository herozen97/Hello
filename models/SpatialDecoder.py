import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules


class SpatialDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # decoder
        dim_in = params.dim_embed_loc
        dim_linear_list = [dim_in] + params.dim_LD_linear + [params.lid_size+3]
        self.loc_decoder = modules.DecoderTF(params, dim_in, params.head_num_loc, params.dim_dec_loc, params.num_dec_loc, params.traj_len+2, dim_linear_list)
        

    def forward(self, loc_tgt, loc_tgt_embed, zs):

        # decode location chain in the next day
        loc_chain = self.loc_decoder(loc_tgt, loc_tgt_embed, zs)

        return loc_chain
        


        