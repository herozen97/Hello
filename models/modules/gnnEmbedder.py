import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GINEConv, GATv2Conv

class gnnEmbedder(nn.Module):
    def __init__(self, params, edge_name='flow'):
        super().__init__()
        # get parameters
        node_info_list, edge_info_list = params.graph_metadata
        # self parameters
        self.relu_type = params.type_relu_gnn
        self.id_contained = params.graph_id_contained
        self.device = params.device
        self.dropout = params.dropout
        # define embedder for id
        self.id_embedder = {}
        self.id_embedder['cell'] = nn.Embedding(params.lid_size+3, params.dim_embed_lid, padding_idx=0).to(params.device)
        self.id_embedder['tract'] = nn.Embedding(params.tract_size, params.dim_embed_lid).to(params.device)
        # define dimension transformer
        dim_node_raw = params.dim_embed_lid + params.dim_graph_node if self.id_contained else params.dim_graph_node
        self.transformer = {}
        for node in node_info_list:
            self.transformer[node] = nn.Linear(dim_node_raw, params.dim_embed_loc).to(params.device)

        # define GNNConv and Batch Norm
        self.convs = nn.ModuleList()
        self.norms = {}
        for node in node_info_list:
            self.norms[node] = nn.ModuleList()
        for idx in range(params.num_gnn_layer):
            # GNNConv
            conv = HeteroConv({
                edge_info: GNN_Factory(params, 
                    params.type_gnn_flow if edge_name in edge_info else params.type_gnn_spa,
                    edge_name in edge_info)
                    for edge_info in edge_info_list
            })
            self.convs.append(conv)
            # BatchNorm
            for node in node_info_list:
                self.norms[node].append(nn.BatchNorm1d(params.dim_embed_loc).to(params.device))

    def forward(self, graph):
        # get data
        x_dict, edge_index_dict, edge_attr_dict = graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict
        if self.id_contained:
            for key, x in x_dict.items():
                id_embed = self.id_embedder[key](torch.arange(x.shape[0]).to(self.device))
                x_dict[key] = torch.concat([x, id_embed], dim=1)
        
        x_dict = {key: self.transformer[key](x) for key, x in x_dict.items()}
        
        # iterate layers
        for idx in range(len(self.convs)):
            x_dict = self.convs[idx](x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.dropout(GNN_ReLU(self.norms[key][idx](x), self.relu_type), p=self.dropout) for key, x in x_dict.items()} 
            
        return x_dict['cell']



def GNN_Factory(params, gnn_type, edge_contain):    
    if gnn_type == 'GAT2':
        assert params.dim_embed_loc % params.num_head_gat == 0, 'Mismatch between head number and out channel '
        return GATv2Conv((-1, -1), params.dim_embed_loc // params.num_head_gat,
                            heads=params.num_head_gat, edge_dim=params.dim_graph_edge if edge_contain else None, add_self_loops=False)
    elif gnn_type == 'SAGE':
        return SAGEConv((-1, -1), params.dim_embed_loc)
    elif gnn_type == 'GAT':
        assert params.dim_embed_loc % params.num_head_gat == 0, 'Mismatch between head number and out channel '
        return GATConv((-1, -1), params.dim_embed_loc // params.num_head_gat, 
                           heads=params.num_head_gat, edge_dim=params.dim_graph_edge if edge_contain else None, add_self_loops=False)
    elif gnn_type == 'GINE':
        if not edge_contain: raise Exception('GINE require edge input')
        return GINEConv(nn.Linear(params.dim_embed_loc, params.dim_embed_loc), edge_dim=params.dim_graph_edge)
    elif gnn_type == 'GIN':
        return gnn.GINConv(nn.Linear(params.dim_embed_loc, params.dim_embed_loc))
    else:
        raise Exception('Unrecognized gnn type')

def GNN_ReLU(x, relu_type):
    if relu_type == 'leaky':
        return F.leaky_relu(x)
    else:
        return F.relu(x)