import os
import time
import argparse
import torch


def param_settings(params=None):
    # Parser Definition
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--path_traj_all', type=str, default='./data/Boston_traj.pkl', help='path of trajectory')
    parser.add_argument('--path_traj', type=str, default='./data/Boston_traj_split.pkl', help='path of trajectory')
    parser.add_argument('--path_attr', type=str, default='./data/Boston_attr.pkl', help='path of ')
    parser.add_argument('--path_mask_day', type=str, default='./data/Boston_mask_day.pkl', help='path of ')
    parser.add_argument('--path_mask_traj', type=str, default='./data/Boston_mask_traj.pkl', help='path of ')
    parser.add_argument('--path_loccoor', type=str, default='./data/Boston_loccoor.pkl', help='path of loc coordinate')
    parser.add_argument('--path_graph', type=str, default='./data/Boston_graph.pkl', help='path of graph')
    parser.add_argument('--path_out', type=str, default='./results/', help='output data path')
    parser.add_argument('--path_out_flag', action='store_false', default=True, help='whether to make output data path')
    parser.add_argument('--out_filename', type=str, default='test', help='output data filename')
    parser.add_argument('--traj_len', type=int, default=48, help='max length of trajectory')
    parser.add_argument('--lid_size', type=int, default=10294, help='size of location id')
    parser.add_argument('--win_len', type=int, default=7, help='historical time window (#days)')
    parser.add_argument('--max_mask_day', type=int, default=3, help='max mask day in time window')
    # process params
    parser.add_argument('--gpu', type=str, default=1, help='GPU index to choose')
    parser.add_argument('--seed', type=int, default=66, help='seed of random')
    parser.add_argument('--run_num', type=int, default=1, help='run number')
    parser.add_argument('--epoch_num', type=int, default=30, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer, choice is adam, adagrad, rmsprop')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--evaluate_step', type=int, default=2, help='evaluate step')
    parser.add_argument('--evaluate_start', type=int, default=1, help='evaluate step')
    parser.add_argument('--tensorboard', default=True, action='store_false', help='whether to use tensorboard')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # loss related
    parser.add_argument('--lam_t', type=float, default=1, help='lambda of time loss')
    parser.add_argument('--eos_weight', type=float, default=1, help='weight of eos in CE loss')

    # model params
    # uid embedding
    parser.add_argument('--uid_contain', action='store_false', default=True, help='whether to embed uid')
    parser.add_argument('--dim_embed_uid', type=int, default=40, help='dimension of uid embedding')
    # spatial related
    parser.add_argument('--dim_com_list', type=str, default='[128]', help='dimension list of location compressor')
    parser.add_argument('--dim_LD_linear', type=str, default='[800,4000]', help='dimension list of location decoder')
    parser.add_argument('--head_num_loc', type=int, default=8, help='location-related head number of MultiHeadAttention')
    parser.add_argument('--dim_enc_loc', type=int, default=1024, help='location-related feedforward dimension of encoder')
    parser.add_argument('--dim_dec_loc', type=int, default=1024, help='location-related feedforward dimension of decoder')
    parser.add_argument('--num_enc_loc', type=int, default=2, help='location-related layer number of encoder')
    parser.add_argument('--num_dec_loc', type=int, default=2, help='location-related layer number of decoder')
    # temporal related
    parser.add_argument('--dim_embed_tim', type=int, default=512, help='dimension of time embedding')
    parser.add_argument('--head_num_tim', type=int, default=8, help='time-related head number of MultiHeadAttention')
    parser.add_argument('--dim_enc_tim', type=int, default=512, help='time-related feedforward dimension of encoder')
    parser.add_argument('--dim_dec_tim', type=int, default=512, help='time-related feedforward dimension of decoder')
    parser.add_argument('--num_enc_tim', type=int, default=2, help='time-related layer number of encoder')
    parser.add_argument('--num_dec_tim', type=int, default=2, help='time-related layer number of decoder')
    parser.add_argument('--dim_TD_linear', type=str, default='[128,1]', help='dimension list of time decoder')
    # gnn related
    parser.add_argument('--type_loc_embedder', type=str, default='gnn', help='loc embedder type: gnn or nn')
    parser.add_argument('--tract_size', type=int, default=1126, help='dimension of tract embedding')
    parser.add_argument('--dim_embed_lid', type=int, default=256, help='dimension of location id embedding')
    parser.add_argument('--dim_embed_loc', type=int, default=512, help='dimension of location embedding')
    parser.add_argument('--dim_graph_node', type=int, default=24, help='dimension of raw node feature')
    parser.add_argument('--dim_graph_edge', type=int, default=24, help='dimension of raw edge feature')
    parser.add_argument('--type_gnn_flow', type=str, default='GAT', help='GNN type of flow map')
    parser.add_argument('--type_gnn_spa', type=str, default='SAGE', help='GNN type of spatial map')
    parser.add_argument('--type_relu_gnn', type=str, default='leaky', help='ReLU type in GNN')
    parser.add_argument('--num_gnn_layer', type=int, default=2, help='gnn layer number')
    parser.add_argument('--num_head_gat', type=int, default=8, help='head number in GAT')
    parser.add_argument('--graph_id_contained', action='store_false', default=True, help='whether node feature contains id embedding')

    # From parser to params
    if params == None:
        params = parser.parse_args()
    else:
        params = parser.parse_args(params)

    # Requirement
    assert params.dim_embed_loc % params.head_num_loc == 0, 'dim_embed_loc must be divisible by head_num_loc'
    assert params.dim_embed_tim % params.head_num_tim == 0, 'dim_embed_tim must be divisible by head_num_tim'

    # File Settings
    if params.path_out_flag:
        params.path_out = f'{params.path_out}{time.strftime("%Y%m%d%H")}_{params.out_filename}/'

    # Parameter adjust
    params.dim_com_list = eval(params.dim_com_list)
    params.dim_LD_linear = eval(params.dim_LD_linear)
    params.dim_TD_linear = eval(params.dim_TD_linear)
    params.dim_graph_node = params.traj_len
    params.dim_graph_edge = params.traj_len


    # General Settings
    # make dir
    if not os.path.exists(params.path_out):
        os.makedirs(params.path_out)
    # gpu initial
    if params.gpu == '-1':
        params.device = torch.device("cpu")
    else:
        params.device = torch.device("cuda", int(params.gpu))
        
    return params
