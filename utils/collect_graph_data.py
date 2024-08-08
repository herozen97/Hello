import pickle
import numpy as np
import torch
import geopandas as gpd
from shapely.geometry.point import Point
import time


def collect_adjecent(loc_coor, tract_info, dist=0.01):
    '''
    Output:
        edge_near_cell, edge_near_tract: torch.tensor, shape=(2, edge_num)
    '''
    # near_cell: distance < dist, and cell in tract
    start_time = time.time()
    print('==== Collecting adjacent for cell')
    # define variable
    edge_near_cell = set()
    edge_in = set()
    cell2tract = {}
    # use to compare
    tmp = np.array((dist, dist))
    dist = np.power(dist, 2)
    percent = len(loc_coor) // 100
    for idx, (loc_src, coor_src) in enumerate(loc_coor.items()):
        if idx % percent == 0: print(f'{idx // percent}%', end= ',')
        for loc_tar, coor_tar in loc_coor.items():
            if loc_src == loc_tar: continue
            # a faster judge: one position larger then continue
            src, tar = np.array(coor_src), np.array(coor_tar)
            if (np.abs(src - tar) > tmp).any(): continue
            # a accurate judge
            if np.power(src - tar, 2).sum() < dist:
                edge = (loc_src, loc_tar) if loc_src < loc_tar else (loc_tar, loc_src)
                edge_near_cell.add(edge)
        # tract_in = tract_info[tract_info.contains(Point(coor_src))].index
        # if len(tract_in) > 0: tract_in = tract_in[0]
        # select the closet tract
        geometry = gpd.GeoSeries(Point(coor_src), crs='epsg:4326').to_crs('epsg:3857')
        distance = tract_info.to_crs('epsg:3857').distance(geometry[0])
        distance.sort_values(ascending=True, inplace=True)
        tract_in = distance.index[0]
        # print(distance); raise Exception
        edge_in.add((loc_src, tract_in))
        cell2tract[loc_src] = tract_in
    print(f'\nDone, time cost is {(time.time()-start_time)/60:.1f}mins')

    # near tract: adjacent
    start_time = time.time()
    print('==== Collecting adjacent for tract')
    edge_near_tract = set()
    for src in range(tract_info.shape[0]):
        for tar in range(tract_info.shape[0]):
            if src == tar: continue
            edge = (src, tar) if src < tar else (tar, src)
            edge_near_tract.add(edge)
    # change data form
    edge_in = torch.tensor(np.array(list(edge_in))).mT.long()
    edge_near_cell = torch.tensor(np.array(list(edge_near_cell))).mT.long()
    edge_near_tract = torch.tensor(np.array(list(edge_near_tract))).mT.long()
    print(f'Done, time cost is {(time.time()-start_time)/60:.1f}mins')

    return edge_in, edge_near_cell, edge_near_tract, cell2tract


def collect_from_traj(uid_traj, loc_coor, cell2tract, lid_size, tract_size, traj_len,
                        win_len=7, split=[0.6, 0.1, 0.3]):
    '''
    Output:
        node_cell, node_tract: torch.tensor, shape=(node_num, node_feat)
        edge_flow_*_idx: torch.tensor, shape=(2, edge_num) 
        edge_flow_*_attr: torch.tensor, shape=(edge_num, feat_num)
    '''
    # define variable
    # edge related
    edge_flow_cell = {}
    edge_flow_tract = {}
    # node related 
    node_cell = np.zeros((lid_size+3, traj_len))
    node_tract = np.zeros((tract_size, traj_len))
    # not in tract
    num_out = 0

    # iterate user
    start_time = time.time()
    print('==== Collecting graph from trajectory')
    percent = len(uid_traj) // 100
    for uidx, traj_all in enumerate(uid_traj.values()):
        if uidx % percent == 0: print(f'{uidx // percent}%', end= ',')
        # constrain to train data
        traj_num = len(traj_all['loc'])
        valid_num = traj_num - win_len
        num_train = int(valid_num * split[0])
        # iterate trajectory
        for idx in range(traj_num):
            if idx == num_train: break
            loc_day = traj_all['loc'][idx]
            tim_day = traj_all['tim'][idx]
            tim_last = -1
            # iterate location
            for idx, loc_cell_cur in enumerate(loc_day[1:]):
                # invalid location
                if loc_cell_cur in [0, lid_size+1]: break
                # not in tract
                if loc_cell_cur not in cell2tract: num_out += 1; continue
                # get tract info 
                loc_tract_cur = cell2tract[loc_cell_cur]
                # update node info
                tim_scope = (sum(tim_day[:tim_last+1]), sum(tim_day[:idx+1]))
                node_cell[loc_cell_cur, tim_scope[0]:tim_scope[1]] += 1
                node_tract[loc_tract_cur, tim_scope[0]:tim_scope[1]] += 1
                # update flow edge info
                if idx != 0:
                    flow_cell = (loc_cell_last, loc_cell_cur)
                    flow_tract = (loc_tract_last, loc_tract_cur)
                    if flow_cell not in edge_flow_cell:
                        edge_flow_cell[flow_cell] = np.zeros(traj_len)
                    if flow_tract not in edge_flow_tract:
                        edge_flow_tract[flow_tract] = np.zeros(traj_len)
                    edge_flow_cell[flow_cell][sum(tim_day[:idx])] += 1
                    edge_flow_tract[flow_tract][sum(tim_day[:idx])] += 1
                loc_cell_last = loc_cell_cur
                loc_tract_last = loc_tract_cur
                tim_last = idx
    print(f'\nIterate trajectory finished. #Out tract record is {num_out}')
    if num_out > 0:
        raise Exception('Cell out of tract')

    # process data to tensor
    node_cell = torch.tensor(node_cell).float()
    node_tract = torch.tensor(node_tract).float()
    edge_flow_cell_idx = torch.tensor(np.array(list(edge_flow_cell.keys()))).mT.long()
    edge_flow_tract_idx = torch.tensor(np.array(list(edge_flow_tract.keys()))).mT.long()
    edge_flow_cell_attr = torch.tensor(np.array(list(edge_flow_cell.values()))).float()
    edge_flow_tract_attr = torch.tensor(np.array(list(edge_flow_tract.values()))).float()
    print(f'Done, time cost is {(time.time()-start_time)/60:.1f}mins')

    return node_cell, node_tract, edge_flow_cell_idx, edge_flow_cell_attr, edge_flow_tract_idx, edge_flow_tract_attr

