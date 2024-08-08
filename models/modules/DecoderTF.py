import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules


class DecoderTF(nn.Module):
    def __init__(self, params, dim_in, head_num, dim_feedforward, layer_num, seq_len, dim_linear_list=[]):
        super().__init__()
        # param settings
        self.device = params.device

        # position encoder
        self.pos_encoder = modules.PositionEncoder(dim_in, params.dropout, seq_len)

        # pattern decoder
        decoder_layer = nn.TransformerDecoderLayer(dim_in, head_num, 
                            dim_feedforward, params.dropout, batch_first=True)
        self.pattern_decoder = nn.TransformerDecoder(decoder_layer, layer_num)

        # location decoder
        self.loc_decoder = nn.ModuleList()
        for i in range(len(dim_linear_list)-1):
            self.loc_decoder.append(nn.Linear(dim_linear_list[i], dim_linear_list[i+1]))
            if i < len(dim_linear_list) - 2:
                self.loc_decoder.append(nn.ReLU())

    def forward(self, X_tgt, X_tgt_embed, z_hid):

        assert X_tgt.shape[1] == X_tgt_embed.shape[1]

        # generate mask
        tgt_mask, tgt_key_padding_mask = self.create_tgt_mask(X_tgt)
        # position encoder
        X_tgt_embed = self.pos_encoder(X_tgt_embed)
        # decode with transformerDeocder and Linear
        X_out = self.pattern_decoder(X_tgt_embed, z_hid.unsqueeze(1), tgt_mask=tgt_mask, 
                                     tgt_key_padding_mask=tgt_key_padding_mask)   # X_out:(B,S,E), X_tgt_embed:(B,S,E), z_hid:(B,H)
        for layer in self.loc_decoder:
            X_out = layer(X_out)   # X_out:(B, S, L)
        return X_out
        

    def create_tgt_mask(self, X_tgt):
        tgt_seq_len = X_tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        tgt_key_padding_mask = (X_tgt == 0)

        return tgt_mask, tgt_key_padding_mask

    def generate_square_subsequent_mask(self, sz: int):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1)
