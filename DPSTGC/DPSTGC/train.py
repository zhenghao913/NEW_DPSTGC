#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.DPSTGC import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn, get_test_new,predict_main_new
from tensorboardX import SummaryWriter
from lib.metrics import masked_mae
from lib.utils_edge import get_directed_matrix
from lib.utils_edge import get_directed_matrix, get_vertices_correlation, Get_new_adj
from lib.utils import norm_Adj
from lib.utils_edge import get_mul_sungraph_adjacency_matrix
from collections import defaultdict
from lib.utils import plt_history


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS08.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
static_delay_matrix_filename = data_config['static_delay_matrix_filename']

if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours #1
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
e_p = int(training_config['e_p'])
e_g = int(training_config['e_g'])
nid = list(map(int, training_config["nid"].split(",")))
heads = int(training_config['heads'])
kt = 3
nb_tempo_filter = 32
q=8
v=8
h=8
N=1
attention_size=None
dropout=0.3
chunk_mode=None
pe = None
pe_period=6

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std, _min, _max= load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

#得到邻接矩阵
adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).to(torch.float32)
#有向图编码
get_directed_matrix = get_directed_matrix(adj_filename, num_of_vertices)
_from, _to, _distances= get_vertices_correlation(adj_filename, num_of_vertices)

#模型函数
net = make_model(DEVICE, nb_block, in_channels, kt, K, nb_chev_filter,nb_tempo_filter, e_g, e_p, nid, heads, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices)
#得到关于延迟相关性的静态矩阵
static_delay_matrix = np.load(static_delay_matrix_filename)
static_delay_matrix = torch.from_numpy(static_delay_matrix).to(torch.float32)
def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    #
    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    history = defaultdict(list)
    train_loss_list = []
    validation_loss_list = []

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, norm_Adj_matrix, static_delay_matrix, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch, _from, _to)
        else:
            val_loss = compute_val_loss_mstgcn(net, norm_Adj_matrix, static_delay_matrix, val_loader, criterion, masked_flag, missing_value, sw, epoch, _from, _to)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)
        validation_loss_list.append(val_loss)

        tmp = []
        net.train()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs, norm_Adj_matrix, static_delay_matrix, _from, _to)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()
            tmp.append(training_loss)
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))
        train_loss_elem = sum(tmp) / len(tmp)
        train_loss_list.append(train_loss_elem)
    print('best epoch:', best_epoch)
    history['train_loss'] = train_loss_list
    history['validation_loss'] = validation_loss_list
    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, _from, _to ,'test')
    plt_history(history, epochs, "DPSTGC")

def predict_main(global_step, data_loader, data_target_tensor, metric_method, _mean, _std, _from, _to, type):

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(net, norm_Adj_matrix, static_delay_matrix, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, _from, _to, type)

    data_loader_new = get_test_new()
    predict_main_new(net, global_step, data_loader_new, get_adjacency_matrix,static_delay_matrix,_from,_to, metric_method, params_path, _max, _min, type)


if __name__ == "__main__":

    train_main()














