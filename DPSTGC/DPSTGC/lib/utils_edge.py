import argparse
import configparser
import numpy as np
import torch
import csv
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']


adj_filename = data_config['adj_filename']
num_of_vertices = int(data_config['num_of_vertices'])

if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
def get_mul_sungraph_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    csv文件：from，to，cost------edges information
    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix(此处返回的是将子图设计为全连接图的邻接矩阵)

    '''
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)
    _from  = []
    _to = []
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            A[i, j] = 1
            distaneA[i, j] = distance
            _from.append(i)
            _to.append(j)
    file = open("static_delay/mul_small_graph/adj_cluster.txt", "r")  # 打开文件
    file_data = file.readlines()
    for row in file_data:
        if row.split() == "":
            continue
        l1 = row.split(" ")
        for i in l1:
            for j in l1:
                i = int(i)
                j = int(j)
                if i ==j:
                    continue
                if A[i, j] != 1:
                    _from.append(i)
                    _to.append(j)
                A[i, j] = 1
    return A, distaneA, _from, _to
#adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
def DSN2(t):
    a = t.sum(dim=1, keepdim=True)
    b = t.sum(dim=0, keepdim=True)
    lamb = torch.cat([a.squeeze(), b.squeeze()], dim=0).max()
    r = t.shape[0] * lamb - t.sum(dim=0).sum(dim=0)

    a = a.expand(-1, t.shape[1])
    b = b.expand(t.shape[0], -1)
    tt = t + (lamb ** 2 - lamb * (a + b) + a * b) / r

    ttmatrix = tt / tt.sum(dim=0)[0]
    ttmatrix = torch.where(t > 0, ttmatrix, t)
    return ttmatrix


def DSN(x):
    """Doubly stochastic normalization"""
    p = x.shape[0]
    y1 = []
    for i in range(p):
        y1.append(DSN2(x[i]))
    y1 = torch.stack(y1, dim=0)
    return y1

def get_directed_matrix(distance_df_filename, num_of_vertices):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx

    else:

        import csv

        A = np.zeros((3, int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[0, i, j] = 1
                A[1, j, i] = 1
                A[2, i, j] = A[0, i, j] + A[1, j, i]
    A = torch.from_numpy(A)
    A = DSN(A)
    return A

#返回列表---from,to,distances
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
                start.append(i)
                end.append(j)
                distances.append(distance)
            print(start)
            print(end)
    return start, end, distances

def get_list_index(_list,x):
    bridge = []
    for i in range(len(_list)):
        if x == _list[i]:
            bridge.append(i)  # 这个end[i]作为from时所有end下标
            return bridge
#返回新的图结构信息
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
        adj_encode = np.zeros((3, int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        for k in range(len(_from)):
            i, j = int(_from[k]), int(_end[k])
            adj_encode[0, i, j] = 1
            adj_encode[1, j, i] = 1
            adj_encode[2, i, j] = adj_encode[0, i, j] + adj_encode[1, j, i]
        adj_encode = torch.from_numpy(adj_encode)
        adj_encode = DSN(adj_encode)
        return adj_encode, A, distaneA, _from, _end
