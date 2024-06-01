import os
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from collections import Counter


graph_signal_matrix_filename = "../data/PEMS07/PEMS07.npz"
num_of_weeks = 0
num_of_days = 0
num_of_hours = 1
batch_size = 32
distance_df_filename = "../data/PEMS07/PEMS07.csv"
adj_filename = "../data/PEMS07/PEMS07.csv"
num_of_vertices = 883
num_for_predict = 12
points_per_hour = 12
l = 16
save_delay = "static_delay.npy"


def data_transformation(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks):
    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    #(26208, 358, 1)
    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
                                num_of_weeks))

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  #3种交通测量量，总流量、平均速度和平均占用率
    #print("train_x", train_x.shape) # train_x (15711, 358, 1, 12)
    train_x = train_x[:, :, 0:1, :]  # [开始位置：终止位置]前闭后开，只剩下一个总流量
    train_target = file_data['train_target']
    train_x = np.squeeze(train_x, axis=2)
    train_x = torch.from_numpy(train_x)
    print("train_x", train_x.shape) #得到torch.Size([15711, 358, 12])
    #舍弃最后不够一小时的点，即15711 - 3 = 15708 = 1309 * 12
    train_x = train_x[:, :, :].reshape(1410, 12, 883, 12)[:, 0, :, :]
    print(train_x.shape)

    #在每个小时内进行聚合(1309, 12, 358, 12)
    train_x = torch.squeeze(train_x)#(1309, 12, 358, 12)
    train_x = train_x.permute(1, 0, 2).reshape(883, -1)

    #得到的数据维度：（307,10176 = 848 * 12）
    return train_x

data_transformation(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks)

#返回原图拓扑信息,列表---from,to,distances
def get_vertices_correlation(distance_df_filename, num_of_vertices):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        start = []
        end = []
        distances = []
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                distaneA[i][j] = distance
                A[i][j] = 1
                start.append(i)
                end.append(j)
                distances.append(distance)
        print(start)
        print(end)
    return start, end, distaneA

def get_list_index(_list,x):
    bridge = []
    for i in range(len(_list)):
        if x == _list[i]:
            bridge.append(i)  # 这个end[i]作为from时所有end下标
            return bridge
#返回新的图结构信息，此处是根据句from,to增加了一些边的图
def Get_new_adj(distance_df_filename, num_of_vertices, distance_bound):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        _from = []
        _end = []
        _distances = []

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            _from.append(i)
            _end.append(j)
            _distances.append(distance)
            A[i, j] = 1
            distaneA[i, j] = distance
        d = 1
        while d:
            edge_num = len(_from)
            d = 0
            for i in range(edge_num):#遍历当前所有边
                if _end[i] in _from:
                    bridge = get_list_index(_from, _end[i])
                    #print(len(bridge))
                    for k in bridge:
                        if _distances[i] + _distances[k] <= distance_bound and A[_from[i],_end[k]] ==0:#如果当前二阶邻居小于阈值，增加为一阶邻居
                            A[_from[i], _end[k]] = 1
                            d = 1
                            _from.append(_from[i])
                            _end.append(_end[k])
                            distaneA[_from[i], _end[k]] = _distances[i] + _distances[k]
                            _distances.append(distaneA[_from[i], _end[k]])
        print("距离阈值为：{0}，边数为：{1}".format(distance_bound, len(_from)))

        return _from, _end, distaneA

#计算延迟相关性
def compulate_delay_correlation(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, _from, _end, distaneA, save_delay):
    #获取训练集数据
    data = data_transformation(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks)
    print("图信号矩阵大小:", data.shape)
    e = np.zeros((2, num_of_vertices, num_of_vertices))#e是2*n*n的矩阵存储延迟系数（-1-1）和延迟长度（0-5）- 6个值
    corr_li = []
    n_910 = 0
    n_89 = 0
    n_78 = 0
    n_67 = 0
    n_56 = 0
    n_05 = 0
    for i in range(len(_from)):  #原始数据中边的个数是340,计算的是所有相连的边的延迟相关性
        #先对延迟长度1-6的延迟相关系数进行遍历，取最相关的延迟长度及其对应的延迟系数（取得是绝对值最大的）
        # 初始化延迟系数及延迟长度
        max_cor = 0
        lag = 0
        for j in range(1, l):
            pearson = np.corrcoef(data[_from[i], j:], data[_end[i], :-j])[0][1]
            if pearson > max_cor:
                max_cor = pearson
                lag = j+1
        #这部分应该是单独计算延迟长度为0时的延迟系数
        pearson = np.corrcoef(data[_from[i], :], data[_end[i], :])[0][1]
        if pearson > max_cor:
            max_cor = pearson
            lag = 1
        pearson = max_cor
        if pearson > 0.9:
            n_910 += 1
        elif pearson > 0.8 and pearson <= 0.9:
            n_89 += 1
        elif pearson > 0.7 and pearson <= 0.8:
            n_78 += 1
        elif pearson > 0.6 and pearson <= 0.7:
            n_67 += 1
        elif pearson > 0.5 and pearson <=0.6:
            n_56 += 1
        else:
            n_05 += 1
        e[0, _from[i], _end[i]] = float(max_cor)
        e[1, _from[i], _end[i]] = int(lag)
        corr_li.append(float(max_cor))
    sum = 0
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):#节点 6 节点 83 延迟长度为: 23.0 延迟系数为: 0.2935718148446808
            # if e[1][i][j] < 10 and e[1][i][j]>3:
            #     print("节点", i, "节点", j, "延迟长度为:", e[1][i][j], "延迟系数为:", e[0][i][j], "节点距离", distaneA[i][j])
            if distaneA[i][j] > 600:
                print(e[1][i][j]-1, end=" ")
                sum+=1
    print("共", sum, "对大于600的边")
    #因为原来延迟相关性在延迟长度为0时，进行one_hot编码，延迟长度为0，numpy初始化为0，这样就分不清除谁是初始化的0，谁是原始的0
    # 所以可以采用延迟长度为lag,实际用lag+1，因为在训练时，已经转化成了one_hot编码，所以真实的延迟长度是几就不重要了。
    print("0.9-1:", n_910, "0.8-0.9:", n_89, "0.7-0.8:", n_78, "0.6-0.7:", n_67, "0.5-0.6:", n_56, "0-0.5:", n_05)
    print("延迟相关系数的个数即，边的个数", np.count_nonzero(e[0]))
    e_statics = e[1]
    unique, count = np.unique(e_statics, return_counts=True)
    data_count = dict(zip(unique, count))
    print("延迟长度的编码信息统计", data_count)
    print("延迟系数：", corr_li)
    e = torch.from_numpy(e)
    num_class = l+1
    lag_code = e[1]
    one_hot = F.one_hot(lag_code.to(torch.int64), num_classes=num_class)  # (N, N, 6)
    # e是2，N,N，需要将二者沿第一个维度合并
    one_hot = one_hot.permute(2, 0, 1)[1:]  # (6, N, N)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!把全为1的第一个通道一个删除了
    print("要合并的两个矩阵形状，一个是延迟系数，一个是延迟长度编码", e[0].shape, one_hot.shape)
    e = torch.cat((torch.unsqueeze(e[0], 0), one_hot), 0)
    print("合并后：", e.shape)
    e_np = e.numpy()
    arr = e[1:]
    print("延迟长度编码形状", arr.shape)
    print("将静态延迟相关性计算结果进行存储，存储为.npy形式，文件名：{0},形状为：{1}".format(save_delay, e_np[0].shape))
    np.save(save_delay, e_np[0])
    print("OK!")
    return


_from, _end, distaneA = get_vertices_correlation(distance_df_filename, num_of_vertices)
compulate_delay_correlation(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, _from, _end, distaneA, save_delay)

