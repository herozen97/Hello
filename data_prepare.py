import pickle
import time
import random
import itertools
import argparse
import geopandas as gpd

import utils



def get_lid_map(path_traj, path_lid_map):

    print('==== Getting lid map')
    # load data
    uid_traj = pickle.load(open(path_traj, 'rb'))
    # get unique location
    loc_list = set()
    percent = len(uid_traj) // 100
    for idx, (uid, traj_all) in enumerate(uid_traj.items()):
        if idx % percent == 0:
            print(idx, end=', ')
        for traj_daily in traj_all:
            # if traj_daily == None: continue
            if traj_daily[0] == 0: continue  # empty record
            loc_chain = [key for key, _ in itertools.groupby(traj_daily)]
            for loc in loc_chain:
                loc_list.add(loc)
                
    # get continuous lid map
    loc_list = list(loc_list)
    loc_list.sort()
    lid_map = {}
    for loc in loc_list:
        lid_map[loc] = len(lid_map) + 1

    # save
    pickle.dump(lid_map, open(path_lid_map, 'wb'))
    print('Done')
    new_ID_list = list(lid_map.values())
    print(f'Total cell is {len(lid_map)}')
    print(f'New cell ID is from {min(new_ID_list)} to {max(new_ID_list)}')
                
                

def update_lid_traj(path_traj, path_lid_map, path_save, path_mask, traj_len=24):
    
    # load data
    start_time = time.time()
    uid_traj_raw = pickle.load(open(path_traj, 'rb'))
    lid_map = pickle.load(open(path_lid_map, 'rb'))
    
    # update trajectory
    uid_traj = {}
    uid_mask = {}
    for uid, traj_all in uid_traj_raw.items():
        uid_traj[uid] = []
        uid_mask[uid] = []
        for traj_daily in traj_all:
            # if traj_daily == None:
            if traj_daily[0] == 0:          # empty record
                uid_traj[uid].append([0] * traj_len)            # This make no sense
                uid_mask[uid].append(1)
                continue
            else:
                uid_traj[uid].append([lid_map[loc] for loc in traj_daily])
                uid_mask[uid].append(0)
    # save
    pickle.dump(uid_traj, open(path_save, 'wb'))
    pickle.dump(uid_mask, open(path_mask, 'wb'))
    print(f'Length of uid_traj: {len(uid_traj)}')
    print('Done')
    print(f'Time cost is {(time.time()-start_time)/60:.1f}mins')


def update_lid_attr(path_attr, path_lid_map, path_save):
    
    # load data
    start_time = time.time()
    uid_attr_raw = pickle.load(open(path_attr, 'rb'))
    lid_map = pickle.load(open(path_lid_map, 'rb'))
    
    # update attr
    uid_attr = {}
    for uid, attr_raw in uid_attr_raw.items():
        attr_new = attr_raw.copy()
        attr_new['home'] = lid_map[attr_raw['home']]
        attr_new['work'] = lid_map[attr_raw['work']] if attr_raw['work'] != None else None
        uid_attr[uid] = attr_new
    # save
    pickle.dump(uid_attr, open(path_save, 'wb'))
    print(f'Length of uid_attr: {len(uid_attr)}')
    print('Done')
    print(f'Time cost is {(time.time()-start_time)/60:.1f}mins')


def update_lid_coor(path_coor, path_lid_map, path_save):
    
    # load data
    start_time = time.time()
    loc_coor_raw = pickle.load(open(path_coor, 'rb'))
    lid_map = pickle.load(open(path_lid_map, 'rb'))
    
    # update attr
    loc_coor = {}
    for loc, coor in loc_coor_raw.items():
        if loc in lid_map:
            loc_coor[lid_map[loc]] = coor
    # save
    pickle.dump(loc_coor, open(path_save, 'wb'))
    print(f'Length of loc_coor: {len(loc_coor)}')
    print(f'New cell ID is from {min(loc_coor.keys())} to {max(loc_coor.keys())}')
    print('Done')
    print(f'Time cost is {(time.time()-start_time)/60:.1f}mins')


def split_traj(path_traj, path_mask_day, path_traj_split, path_mask_traj, token, traj_len=24):
    # load data
    start_time = time.time()
    uid_traj = pickle.load(open(path_traj, 'rb'))
    uid_mask_day = pickle.load(open(path_mask_day, 'rb'))
    
    # split location and duration chain
    uid_traj_split = {}
    uid_mask_traj = {}
    for uid, traj_all in uid_traj.items():
        uid_traj_split[uid] = {'loc': [], 'tim': []}
        uid_mask_traj[uid] = []
        for idx, traj_day in enumerate(traj_all):
            if uid_mask_day[uid][idx]:
                # invalid traj
                uid_traj_split[uid]['loc'].append([0] * (traj_len+2))
                uid_traj_split[uid]['tim'].append([0] * (traj_len))
                uid_mask_traj[uid].append([0] * 2 + [1] * (traj_len))
            else:
                loc_chain, tim_chain, mask_traj = [], [], []
                for key, group in itertools.groupby(traj_day):
                    loc_chain.append(key)
                    tim_chain.append(len(list(group)))
                    mask_traj.append(0)
                delta_num = traj_len - len(loc_chain)           # traj_len+2 includes SOS and EOS
                uid_traj_split[uid]['loc'].append([token.SOS] + loc_chain + [token.EOS] + [0] * delta_num)   # 26   
                uid_traj_split[uid]['tim'].append(tim_chain + [0] * delta_num)              # 24
                uid_mask_traj[uid].append([0] + mask_traj + [0] + [1] * delta_num)          # 26

    # save
    pickle.dump(uid_traj_split, open(path_traj_split, 'wb'))
    pickle.dump(uid_mask_traj, open(path_mask_traj, 'wb'))
    print('Done')
    print(f'Time cost is {(time.time()-start_time)/60:.1f}mins')

    
    
def construct_graph(path_save_traj_split, path_save_loccoor, path_save_graph, params):
    # load data
    uid_traj = pickle.load(open(path_save_traj_split, 'rb'))
    loc_coor = pickle.load(open(path_save_loccoor, 'rb'))
    town_info = gpd.read_file(params.path_town)['geometry']
    print(f'#Loc in loc_coor: {len(loc_coor)}, #Town in town_info: {len(town_info)}')
    # get data
    edge_in, edge_near_cell, edge_near_town, cell2town = utils.collect_adjecent(loc_coor, town_info)
    node_cell, node_town, edge_flow_cell_idx, edge_flow_cell_attr, edge_flow_town_idx, edge_flow_town_attr = utils.collect_from_traj(uid_traj, loc_coor, cell2town, params.lid_size, town_info.shape[0], params.traj_len)
    # store data
    data_store = {
        'cell2town': cell2town,
        'edge_in': edge_in, 'edge_near_cell': edge_near_cell, 'edge_near_town': edge_near_town, 
        'node_cell': node_cell, 'node_town': node_town,
        'edge_flow_cell_idx': edge_flow_cell_idx, 'edge_flow_cell_attr': edge_flow_cell_attr, 
        'edge_flow_town_idx': edge_flow_town_idx, 'edge_flow_town_attr': edge_flow_town_attr}
    pickle.dump(data_store, open(path_save_graph, 'wb'))


            
    

def settings(param=[]):
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--path_traj', type=str, default='None')
    parser.add_argument('--path_attr', type=str, default='None')
    parser.add_argument('--path_loccoor', type=str, default='None')
    parser.add_argument('--path_town', type=str, default='None')
    parser.add_argument('--process', type=str, default='none')
    parser.add_argument('--path_out', type=str, default='./data/')
    parser.add_argument('--file_name', type=str, default='CDRsh')
    parser.add_argument('--lid_size', type=int, default=0)
    parser.add_argument('--traj_len', type=int, default=0)

    if __name__ == '__main__' and param == []:
        params =  parser.parse_args()
    else:
        params = parser.parse_args(param)
        
    return params


if __name__ == '__main__':

    params = settings()
    # save path
    path_lid_map = f'{params.path_out}/{params.file_name}_lid_map.pkl'
    path_save_traj = f'{params.path_out}/{params.file_name}_traj.pkl'
    path_save_attr = f'{params.path_out}/{params.file_name}_attr.pkl'
    path_save_loccoor = f'{params.path_out}/{params.file_name}_loccoor.pkl'
    path_save_traj_split = f'{params.path_out}/{params.file_name}_traj_split.pkl'
    path_mask_day = f'{params.path_out}/{params.file_name}_mask_day.pkl'
    path_mask_traj = f'{params.path_out}/{params.file_name}_mask_traj.pkl'
    path_save_graph = f'{params.path_out}/{params.file_name}_graph.pkl'
    # process
    if params.process == 'lid_map':
        get_lid_map(params.path_traj, path_lid_map)
    elif params.process == 'update_traj':
        update_lid_traj(params.path_traj, path_lid_map, path_save_traj, path_mask_day, params.traj_len)
    elif params.process == 'update_attr':
        update_lid_attr(params.path_attr, path_lid_map, path_save_attr)
    elif params.process == 'update_loccoor':
        update_lid_coor(params.path_loccoor, path_lid_map, path_save_loccoor)
    elif params.process == 'split_traj':
        token = utils.TOKEN(params.lid_size)
        split_traj(path_save_traj, path_mask_day, path_save_traj_split, path_mask_traj, token, params.traj_len)
    elif params.process == 'construct_graph':
        construct_graph(path_save_traj_split, path_save_loccoor, path_save_graph, params)