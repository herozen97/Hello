import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules


class TemporalEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        # parameters
        self.win_len = params.win_len   # default=7
        self.device = params.device
        
        # time embedder
        self.tim_embedder = modules.nnEmbedder(params.traj_len+1, params.dim_embed_tim)
        
        # day encoder
        self.day_encoder = modules.EncoderTF(params, params.dim_embed_tim, params.head_num_tim, params.dim_enc_tim, params.num_enc_tim, params.traj_len)

        # week encoder
        self.week_encoder = modules.EncoderTF(params, params.dim_embed_tim, params.head_num_tim, params.dim_enc_tim, params.num_enc_tim, params.traj_len)


    def forward(self, tim_batch, mask_day, mask_traj):

        tim_day_all = torch.tensor([]).to(self.device)
        # encode daily trajectory
        for i in range(self.win_len):
            # time embedding
            tim_embed = self.tim_embedder(tim_batch[:, i, :])       # tim_batch:(B, 7, S=24), tim_embed:(B,S,E)
            # encode daily trajectory in week
            tim_day = self.day_encoder(tim_embed, mask_traj[:, i, 1:-1]).unsqueeze(1)    # tim_day:(B, 1, H), mask_traj:(B,7,S=26)
            # for k in range(tim_day.size()[0]):
            #     print('k',tim_day[k])
            tim_day_all = torch.cat([tim_day_all, tim_day], dim=1)

        tim_week = self.week_encoder(tim_day_all, mask_day)            # tim_week:(B, H)

        return tim_week
    