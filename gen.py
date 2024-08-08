import time
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
# self defined
import utils
from tqdm import tqdm
import pickle
import torch
import numpy as np

# self defined
from models import MSTDP as Model
import utils
import random
import os



def train(params):
    # load dataset
    print('=' * 20, ' Loading data')
    start_time = time.time()
    uid_traj = pickle.load(open(params.path_traj, 'rb'))
    uid_attr = pickle.load(open(params.path_attr, 'rb'))
    uid_mask_day = pickle.load(open(params.path_mask_day, 'rb'))
    uid_mask_traj = pickle.load(open(params.path_mask_traj, 'rb'))
    loc_coor = pickle.load(open(params.path_loccoor, 'rb'))
    # graph data
    if params.type_loc_embedder == 'gnn':
        graph = utils.construct_graph(params.path_graph, params.device)
        params.graph_metadata = graph.metadata()
    print(f'Loading finished! Time cost is {time.time() - start_time:.1f} s')
    # dataset related settings
    params.num_user = len(uid_traj) + 1  # (start from 1)
    params.TOKEN = utils.TOKEN(params.lid_size)

    # initiate model and optimizer
    model = Model(params).to(params.device)

    if params.type_loc_embedder == 'gnn':
        # initiate graph part
        with torch.no_grad():  # Initialize lazy modules.
            _ = model.spatial_encoder.graph_embedder(graph)

    params.optimizer = params.optimizer.lower()
    if params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate,
                                     weight_decay=params.weight_decay)
    elif params.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params.learning_rate,
                                        weight_decay=params.weight_decay)
    elif params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=params.learning_rate,
                                        weight_decay=params.weight_decay)

    try:
        print('=== Model info\n')
        print(model)
        model.count_params()
    except:
        print('Error in counting model info')
    print('=== Optimizer info\n', optimizer)

    # collect best information
    info_best = {'compare': [], 'model_params': None, 'metrics': None, 'epoch': 0}

    # construct batch list
    print('==== Generating batch data')
    start_time = time.time()
    batch_all, uid_map = utils.get_batch_info(uid_traj, params.win_len, uid_mask_day)
    print(f'Time cost to generate batch list is {(time.time() - start_time) / 60:.2f} mins')
    # training
    for epoch in range(1, params.epoch_num + 1):

        model.train()

        # variable
        loss_all = 0.
        loss_s_all = 0.
        loss_t_all = 0.
        batch_num = 0
        # train with batch
        print('=' * 10, f' [Epoch={epoch}]')
        train_time = time.time()
        for data_src, data_tar in utils.get_data_batch(uid_traj, uid_attr, uid_mask_day, uid_mask_traj,
                                                       uid_map,
                                                       batch_all['train'], params):
            # data_src=(uid_batch, attr_batch, loc_src, tim_src, mask_day, mask_traj), data_tar=(loc_tar, tim_tar)
            # forward
            if params.type_loc_embedder == 'gnn':
                data_sync = model((data_src, graph), data_tar[0])
            else:
                data_sync = model(data_src, data_tar[0])
            loss_s, loss_t = model.calculate_loss(data_sync, data_tar)
            loss = loss_s + params.lam_t * loss_t
            if loss.isnan():
                raise Exception('Training Error, loss appear nan')

            # reocrd loss
            loss_all += loss.item()
            loss_s_all += loss_s.item()
            loss_t_all += loss_t.item()
            batch_num += 1
            # backwards
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        loss_all /= batch_num
        loss_s_all /= batch_num
        loss_t_all /= batch_num

        print(f'\nTrain time cost is {(time.time() - train_time) / 60:.1f} mins')
        print(f'Loss: all={loss_all:.3f}, loc={loss_s_all:.3f}, tim={loss_t_all:.3f}')
        torch.save(model.state_dict(), f'{params.path_out}model_checkpoint.pth')

        # record with tensorboard writer
        if params.writer != None:
            params.writer.add_scalar('Loss/Loss_all', loss_all, epoch)
            params.writer.add_scalar('Loss/Loss_loc', loss_s_all, epoch)
            params.writer.add_scalar('Loss/Loss_tim', loss_t_all, epoch)


        # evaluation in epoch
        if (epoch >= params.evaluate_start and epoch % params.evaluate_step == 0) or epoch == params.epoch_num or epoch==1:
            print('== Evaluation with validation data')
            start_time = time.time()
            if params.type_loc_embedder == 'gnn':
                dataset = ((uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor), graph)
            else:
                dataset = (uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor)
            metric_valid = evaluate(params, dataset, model, mode='test', epoch=epoch)
            print(f'\nEvaluation time cost: {(time.time() - start_time) / 60:.1f}mins')
            print('Metrics 7th day:', metric_valid)
            if params.writer != None:
                params.writer.add_scalar('Val/Day_Loc_ACC', metric_valid[0], epoch)
                params.writer.add_scalar('Val/Day_Distance_Error', metric_valid[1], epoch)
                params.writer.add_scalar('Val/Day_JSD_TravelDist', metric_valid[2], epoch)
                params.writer.add_scalar('Val/Day_JSD_Departure', metric_valid[3], epoch)

            # store info: half metrics are better then store
            metrics_epoch = np.array(metric_valid)
            # Empty or Better then update
            # check_better(info_best['compare'], metrics_epoch, [1, 0, 0, 0])
            if len(info_best['compare']) == 0 or (int(info_best['compare'][0] < metrics_epoch[0]) + (
                    info_best['compare'][1:] > metrics_epoch[1:]).sum()) > len(metrics_epoch) // 2:
                info_best['compare'] = metrics_epoch
                info_best['model_params'] = model.state_dict()
                info_best['metrics'] = metric_valid
                info_best['epoch'] = epoch
            if epoch == params.epoch_num:
                info_final = {'compare': metrics_epoch,
                              'model_params': model.state_dict(),
                              'metrics': metric_valid}
            # early stop
            if epoch > 0.5 * params.epoch_num and info_best['compare'][0] < 0.1:
                print('Early stop! The parameter settings cannot work!')
                info_final = {'compare': metrics_epoch,
                              'model_params': model.state_dict(),
                              'metrics': metric_valid}
                break

    # save results
    pickle.dump(info_best['metrics'], open(f'{params.path_out}metrics_val_best_week_{params.run_idx}.pkl', 'wb'))
    pickle.dump(info_final['metrics'], open(f'{params.path_out}metrics_val_final_week_{params.run_idx}.pkl', 'wb'))
    torch.save(info_final['model_params'], f'{params.path_out}model_final_week_{params.run_idx}.pth')
    torch.save(info_best['model_params'], f'{params.path_out}model_best_week_{params.run_idx}.pth')
    # generate test data
    print('== generate test data with final')
    # settings
    start_time = time.time()
    if params.type_loc_embedder == 'gnn':
        dataset = ((uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor), graph)
    else:
        dataset = (uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor)

    result = evaluate(params, dataset, model, mode='test', epoch='final')
    print('next day:',result)

    # generate test data
    print(f'== generate test data with best at Epoch {info_best["epoch"]}')
    # settings
    start_time = time.time()
    if params.type_loc_embedder == 'gnn':
        dataset = ((uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor), graph)
    else:
        dataset = (uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor)

    model.load_state_dict(info_best['model_params'])
    result = evaluate(params, dataset, model, mode='test', epoch='best')
    print('next day:',result)

    return result


def evaluate(params, dataset, model, mode, epoch):
    '''
    mode: valid, test
    '''
    print('== eval day')
    # settings
    model.eval()
    start_time = time.time()
    if params.type_loc_embedder == 'gnn':
        (uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor), graph = dataset
    else:
        uid_traj, uid_attr, uid_mask_day, uid_mask_traj, uid_map, batch_all, loc_coor = dataset
    # get uid map reverse
    uid_map_reverse = dict(zip(uid_map.values(), uid_map.keys()))
    # save related variable
    uid_traj_sync = {}
    uid_traj_true = {}

    # predict daily trajectory with batch
    for data_src, (loc_tar, tim_tar) in utils.get_data_batch(uid_traj, uid_attr, uid_mask_day, uid_mask_traj, 
                                                             uid_map,
                                                             batch_all[mode], params, mode):
        # forward
        with torch.no_grad():
            if params.type_loc_embedder == 'gnn':
                loc_sync, tim_sync = model((data_src, graph), loc_tar, mode)
            else:
                loc_sync, tim_sync = model(data_src, loc_tar, mode)
        loc_sync, tim_sync = loc_sync.tolist(), tim_sync.tolist()
        loc_tar, tim_tar  = loc_tar.tolist(), tim_tar.tolist()

        # process iterately
        uid_batch = data_src[0].tolist()
        for idx, uid in enumerate(uid_batch):
            # map reverse
            uid = uid_map_reverse[uid]
            # init
            if uid not in uid_traj_sync:
                uid_traj_sync[uid] = []
                uid_traj_true[uid] = []
            # construct trajectory
            traj_sync = utils.construct_traj_sync(loc_sync[idx], tim_sync[idx], params.TOKEN, params.traj_len, uid_attr[uid]['home'])
            traj_true = utils.construct_traj_true(loc_tar[idx][1:-1], tim_tar[idx], params.TOKEN, params.traj_len)  # loc_tar:(S=26)
            # collect data
            uid_traj_sync[uid].append(traj_sync)
            uid_traj_true[uid].append(traj_true)
    path_out = params.path_out
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    pickle.dump(uid_traj_sync, open(f'{path_out}results_sync_ep{epoch}.pkl', 'wb'))
    pickle.dump(uid_traj_true, open(f'{path_out}results_true_ep{epoch}.pkl', 'wb'))
    print('eval save')

    print(f'Time cost to predict with {mode} data is {(time.time()-start_time)/60:.1f}mins')

    # metrics
    start_time = time.time()
    metrics = utils.cal_metrics(uid_traj_sync, uid_traj_true, loc_coor)
    print('Eval Next Day:', metrics)
    print(f'Time cost to evaluate is {(time.time()-start_time)/60:.1f}mins')

    return metrics

if __name__ == '__main__':

    print('=' * 20, ' Program Start')
    # settings
    # params intial
    params = utils.param_settings()
    print('==== Parameter settings')
    for k, v in params.__dict__.items():
        print(k, v)
    pickle.dump(params.__dict__, open(f'{params.path_out}/parameter.pkl', 'wb'))

    # train with multi-runs
    for run_idx in range(params.run_num):
        params.run_idx = (run_idx + 1)
        # supervise time
        run_time = time.time()
        time_now = (datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        print('=' * 10, f' Run {run_idx + 1}.   Current time is {time_now}')
        # tensorboard writer
        if params.tensorboard:
            params.writer = SummaryWriter(f'{params.path_out}{run_idx}')
        else:
            params.writer = None
        # train
        train(params)
        print(f'This run finished! Time cost is {(time.time() - run_time) / 60:.1f}mins')
