import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules

class SpatialEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        # parameters
        self.win_len = params.win_len   # default=7
        self.device = params.device
        self.type_loc_embedder = params.type_loc_embedder
        
        # location embedder
        if params.type_loc_embedder == 'nn':
            self.loc_embedder = modules.nnEmbedder(params.lid_size+3, params.dim_embed_loc)
        elif params.type_loc_embedder == 'gnn':
            self.graph_embedder = modules.gnnEmbedder(params)
            self.loc_embedder = None
        
        # day encoder
        self.day_encoder = modules.EncoderTF(params, params.dim_embed_loc, params.head_num_loc, params.dim_enc_loc, params.num_enc_loc, params.traj_len+2)

        # week encoder
        self.week_encoder = modules.EncoderTF(params, params.dim_embed_loc, params.head_num_loc, params.dim_enc_loc, params.num_enc_loc, params.traj_len+2)


    def forward(self, loc_batch, mask_day, mask_traj):

        loc_day_all = torch.tensor([]).to(self.device)
        # encode daily trajectory
        for i in range(self.win_len):
            # location embedding
            if self.type_loc_embedder == 'gnn':
                loc_embed = self.loc_embedder[loc_batch[:, i, :]]    # loc_batch:(B, 7, S), loc_embed:(B, S, E)
            else:
                loc_embed = self.loc_embedder(loc_batch[:, i, :]) 
            loc_day = self.day_encoder(loc_embed, mask_traj[:, i, :]).unsqueeze(1)   # loc_day:(B, 1, E), mask_traj:(B,7,S)
            loc_day_all = torch.cat([loc_day_all, loc_day], dim=1)  
        
        # encode whole week infomation
        loc_week = self.week_encoder(loc_day_all,mask_day)   # loc_day_all:(B, 7, E), loc_week:(B, E)

        return loc_week, loc_day_all
        