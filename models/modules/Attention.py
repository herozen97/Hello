import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()


        self.V = nn.Linear(dim_hidden, 1) 
        self.Linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.Linear2 = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, h, hc):
        # h:(N,L,h)
        # get attention
        atten = self.V(F.tanh(self.Linear1(hc) + self.Linear2(h)))   # atten:(N, L, 1)
        atten = atten.transpose(1, 2)       # atten:(N,1,L)
        weight = F.softmax(atten, dim=-1)
        # get weighted hidden
        hidden = torch.bmm(weight, h)   # (N,1,L)*(N,L,h)->(N,1,h)

        return hidden  
