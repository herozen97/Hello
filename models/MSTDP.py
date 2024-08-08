import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# self-defined
import models
from models import modules



class MSTDP(nn.Module):
    def __init__(self, params):
        super().__init__()
        # parameter settings
        self.device=params.device
        self.lid_size = params.lid_size
        self.traj_len = params.traj_len
        self.type_loc_embedder = params.type_loc_embedder
        self.uid_contain = params.uid_contain
        
        # uid embedder
        if params.uid_contain:
            self.uid_embedder = nn.Embedding(params.num_user, params.dim_embed_uid)
            self.hid_mapper = nn.Linear(params.dim_embed_uid+params.dim_embed_loc, params.dim_embed_loc)

        # encoder
        self.spatial_encoder = models.SpatialEncoder(params)
        self.temporal_encoder = models.TemporalEncoder(params)
        
        # decoder
        self.spatial_decoder = models.SpatialDecoder(params)
        self.temporal_decoder = models.TemporalDecoder(params)

        # compressor
        dim_compress = [params.dim_embed_loc] + params.dim_com_list
        self.loc_compressor = modules.LocCompressor(dim_compress)

        # weights of stop/pad to calculate loss_s
        self.loss_s_weight = torch.ones(params.lid_size+3)
        self.loss_s_weight[-2] = params.eos_weight  # EOS
        self.loss_s_weight = self.loss_s_weight.to(params.device)

        

    def forward(self, data_batch, loc_tar, mode='train'):

        # get data
        if self.type_loc_embedder == 'gnn':
            (uid_batch, _, loc_src, tim_src, mask_day, mask_traj), graph = data_batch
        else:
            # attr_batch=3*(B,1), loc_src=(B,7,S=26) with token, tim_src=(B,7,S=24), mask_day=(B, 7), mask_traj=(B,7,S=26), loc_tar=(B,S)
            uid_batch, _, loc_src, tim_src, mask_day, mask_traj = data_batch
        # if use gnnEmbedder, get loc_map
        if self.type_loc_embedder == 'gnn':
            self.spatial_encoder.loc_embedder = self.spatial_encoder.graph_embedder(graph)
            self.spatial_encoder.loc_embedder = self.spatial_encoder.loc_embedder

        # encode trajectory
        zs, hid_loc = self.spatial_encoder(loc_src, mask_day, mask_traj)    # hid_loc: spatial hidden variable for week     # zs:(B, El), hid_loc:(B, 7, E)
        zt = self.temporal_encoder(tim_src, mask_day, mask_traj)   # (B, H)

        if self.uid_contain:
            # embed uid
            uid_embed = self.uid_embedder(uid_batch)    # (B, Eu)
            # concat zs and zuid, then map
            zs = self.hid_mapper(torch.cat([zs, uid_embed], dim=-1))  # (B, Eu+El) -> (B, El)

        # decode trajectory
        # train mode
        if mode == 'train':
            if self.type_loc_embedder == 'gnn':
                loc_tar_embed = self.spatial_encoder.loc_embedder[loc_tar[:, :-1]]       # loc_tar:(B,S=26), (B, S=25, El)
            else:
                loc_tar_embed = self.spatial_encoder.loc_embedder(loc_tar[:, :-1])
            loc_chain_p = self.spatial_decoder(loc_tar[:, :-1], loc_tar_embed, zs)       # loc_chain_p:(B, S=25, L)
            loc_chain_idx = torch.argmax(F.log_softmax(loc_chain_p, dim=-1), dim=-1)     # loc_chain_idx: (B,S=25)
            if self.type_loc_embedder == 'gnn':
                loc_info = self.loc_compressor(self.spatial_encoder.loc_embedder[loc_chain_idx[:, :-1]])  # loc_info: (B, S=24, H)
            else:
                loc_info = self.loc_compressor(self.spatial_encoder.loc_embedder(loc_chain_idx[:, :-1]))
            tim_chain = self.temporal_decoder(zt, loc_info, loc_chain_idx[:, :-1])  # tim_Chain: (B, S=24)
            return (loc_chain_p, tim_chain)
        # eval mode
        else:
            loc_chain = loc_tar[:, :].clone()    # loc_tar:(B,S=26)
            loc_chain[:, 1:] = 0                 # loc_chain:(B,S=26)
            loc_chain_reshaped = loc_chain.reshape(-1, loc_chain.shape[-1])  # loc_chain_reshaped: (B',S)
            zs_reshaped = zs.reshape(-1, zs.shape[-1])       # (B',H)
            for i in range(loc_chain.shape[1]-1):
                if self.type_loc_embedder == 'gnn':
                    loc_embed = self.spatial_encoder.loc_embedder[loc_chain_reshaped] # loc_embed: (B',S, E)
                else:
                    loc_embed = self.spatial_encoder.loc_embedder(loc_chain_reshaped)
                loc_pred = self.spatial_decoder(loc_chain_reshaped, loc_embed, zs_reshaped)   # loc_pred:(B',S,L+3)
                loc_pred = F.log_softmax(loc_pred, dim=-1)   # (B',S,L+3)
                loc_next = torch.argmax(loc_pred, dim=-1)   # (B, S)
                loc_chain_reshaped[:, i+1] = loc_next[:, i]  # (B, S)
            loc_chain = loc_chain_reshaped.reshape(loc_chain.shape[0], -1, loc_chain.shape[-1])[:, 0, 1:-1]  # (B', S=26) -> (B, S=24)
            if self.type_loc_embedder == 'gnn':
                loc_info = self.loc_compressor(self.spatial_encoder.loc_embedder[loc_chain])  # loc_info: (B, S=24, H)
            else:
                loc_info = self.loc_compressor(self.spatial_encoder.loc_embedder(loc_chain))
            tim_chain = self.temporal_decoder(zt, loc_info, loc_chain)  # tim_Chain: (B, S=24)

            return loc_chain, tim_chain

    
    def calculate_loss(self, data_sync, data_tar):
        # get data
        loc_sync, tim_sync = data_sync
        loc_tar, tim_tar = data_tar
        # mask with true length
        mask = tim_tar == 0
        tim_sync = tim_sync.masked_fill(mask, 0)
        # calculate loss
        loss_s = F.cross_entropy(loc_sync.reshape(-1, self.lid_size+3), loc_tar[:, 1:].reshape(-1), 
                                     weight=self.loss_s_weight, ignore_index=0)
        loss_t = F.huber_loss(tim_sync, tim_tar.float(),delta=1)

        return loss_s, loss_t
    
    def count_params(self):
        total_num = sum(param.numel() for param in self.parameters())
        trainable_num = sum(param.numel() for param in self.parameters() if param.requires_grad)
        print(f'== Parameter numbers: total={total_num}, trainable={trainable_num}')


