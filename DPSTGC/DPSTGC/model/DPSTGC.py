# -*- coding:utf-8 -*-
#想要运行时，需要将.to(DEVICE)类似去掉或者给改成cpu，另外可以将batch_size改小一点
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial, norm_Adj
from lib.utils_edge import DSN
import numpy as np
import math
from model.temporal_conv.tempo_conv import temporal_conv_layer

class graph_constructor(nn.Module):
    def __init__(self,num_of_vertices, dim):
        super(graph_constructor, self).__init__()
        self.nnodes = num_of_vertices
        self.emb1 = nn.Embedding(num_of_vertices, dim)
        self.emb2 = nn.Embedding(num_of_vertices, dim)
        self.num_of_vertices = num_of_vertices

    def forward(self, x):
        idx = torch.LongTensor(torch.arange(self.num_of_vertices)).to(x.device)
        nodevec1 = self.emb1(idx)#.unsqueeze(0)
        nodevec2 = self.emb2(idx)#.unsqueeze(0)
        A = torch.relu(torch.matmul(nodevec1, nodevec2.T))
        return F.softmax(A, dim=0)

class GraphLearn(torch.nn.Module):
    """
    Graph Learning Modoel for AdapGL.

    Args:
        num_nodes: The number of nodes.
        init_feature_num: The initial feature number (< num_nodes).
    """

    def __init__(self, DEVICE, num_nodes, init_feature_num):
        super(GraphLearn, self).__init__()
        self.epsilon = 1 / num_nodes * 0.5
        self.beta = torch.nn.Parameter(
            torch.rand(num_nodes).to(DEVICE),
            requires_grad=True
        )

        self.w1 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num)).to(DEVICE),
            requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num)).to(DEVICE),
            requires_grad=True
        )

        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.DEVICE = DEVICE

    def forward(self, adj_mx):
        new_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        #print(new_adj_mx.shape)
        new_adj_mx = torch.relu(new_adj_mx + torch.diag(self.beta))
        adj_mx = adj_mx.to(self.DEVICE)
        attn = torch.sigmoid(self.attn(torch.stack((adj_mx, new_adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = torch.relu(d.view(-1, 1) * new_adj_mx * d - self.epsilon)
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d
        return new_adj_mx

class GraphAttentionLayer_delay(nn.Module):
    def __init__(self, DEVICE,  in_features, out_features, num_of_vertices, num_of_timesteps, e_p, is_last=True):
        super(GraphAttentionLayer_delay, self).__init__()
        self.dropout = 0.4
        self.e_p = e_p
        self.num_of_timesteps = num_of_timesteps
        self.is_last = is_last
        self.in_features = in_features
        if self.is_last:
            self.in_features = self.in_features * self.e_p
        self.out_features = out_features
        self.DEVICE = DEVICE
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features).to(DEVICE))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.FloatTensor(size=(2 * out_features * num_of_timesteps, 1)).to(DEVICE))
        #nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, delay_matrix):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        graph_signal = torch.matmul(x, self.W)  # (b, N, T, F_in)*(F_in, F_out) = (b, N, T, F_out)
        hp_list = []
        for k in range(batch_size):  # 对batch_size进行遍历
            vertices_signal = graph_signal[k, :, :, :].reshape(num_of_vertices, -1)  # (N,F*T)
            e = self.update_directed_edge(vertices_signal).to(self.DEVICE)
            delay_matrix = e * delay_matrix
            delay_matrix = F.softmax(delay_matrix, dim=1)
            hp_prime = torch.matmul(delay_matrix, vertices_signal)
            hp_prime = hp_prime.reshape(num_of_vertices, num_of_timesteps, -1)
            hp_list.append(hp_prime)  # (最终得到的list(B,6,N,T,F))
        hp = torch.stack(hp_list, dim=0)  # 得到tensor(B,6,N,T,F)
        return hp
        # if self.is_last:
        #     outputs = torch.sum(hp, dim=1)
        #     outputs = torch.squeeze(outputs, 1)  # (b, N, T, F)
        #     return outputs
        # else:
        #     outputs = hp.permute(0, 2, 3, 1, 4).reshape(batch_size, num_of_vertices, num_of_timesteps, -1)
        #     # (B,6,N,T,F)--->(B,N,T,6,F)------>(B,N,T,6*F)
        #     return F.elu(outputs), delay_matrix
        #     # (B,N,T,6*F),delay_matrix
    def update_directed_edge(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:(self.out_features * self.num_of_timesteps), :])
        Wh2 = torch.matmul(Wh, self.a[(self.out_features * self.num_of_timesteps):, :])
        #print(Wh1.shape, self.a.shape)
        #arf = 0.3
        Wh1 = torch.tanh(Wh1)
        Wh2 = torch.tanh(Wh2)
        # construct directed
        h1 = torch.matmul(Wh1, Wh2.T)
        h2 = torch.matmul(Wh2, Wh1.T)
        e = torch.tanh((h1 - h2))
        return F.relu(e)


class GAT(nn.Module):
    def __init__(self, DEVICE, in_features, out_features,num_of_vertices, e_p, nid_size, heads, num_of_timesteps):
        super(GAT, self).__init__()
        self.heads = heads
        self.nid_size = nid_size
        self.DEVICE = DEVICE
        # self.fir_att_directed = GraphAttentionLayer_directed(in_features*num_of_timesteps, self.nid_size*num_of_timesteps, e_g, is_last=False)
        # self.fir_att_delay = GraphAttentionLayer_delay(DEVICE, in_features, self.nid_size[0], num_of_timesteps, e_p,
        #                                                is_last=False)
        # # self.out_att_directed = GraphAttentionLayer_directed(self.nid_size*num_of_timesteps, out_features*num_of_timesteps, e_g, is_last=True)
        # self.mid_att_delay = GraphAttentionLayer_delay(DEVICE, self.nid_size[0] * e_p, self.nid_size[1], num_of_timesteps, e_p,
        #                                                is_last=False)
        self.out_att_delay = GraphAttentionLayer_delay(DEVICE, in_features, out_features, num_of_vertices, num_of_timesteps, e_p,
                                                       is_last=True)

    def forward(self, x, static_delay_matrix):
        batch_size, num_of_vertices, in_channels, num_of_timesteps, = x.shape  # (B,N,F,T)
        static_delay_matrix = static_delay_matrix.to(self.DEVICE)
        x = x.permute(0, 1, 3, 2)  # (B,N,T,F)
        #static_delay_matrix = static_delay_matrix[1:] * static_delay_matrix[0]  # (6,N,N)
        # _x1, delay_matrix = self.fir_att_delay(x, static_delay_matrix)
        # _x2, delay_matrix = self.mid_att_delay(_x1, delay_matrix)
        # (b, N, T, F)
        outputs = F.elu(self.out_att_delay(x, static_delay_matrix))
        # outputs = out_x2.reshape(batch_size, num_of_vertices, num_of_timesteps, -1)
        return outputs.permute(0, 1, 3, 2)  # (b,n,f,t)

class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):#对时间维度进行遍历

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)
                #空间上的哈达玛积运算将注意力权重加到Tk（L）空间卷积上
                #spatial_attention = spatial_attention.to(torch.float32).to(self.DEVICE)
                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(B,N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.matmul(graph_signal)  # (B, N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)求和

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

class DPSTGC_block(nn.Module):

    def __init__(self, DEVICE, in_channels,kt, K, nb_chev_filter,nb_tempo_filter,last_filter, e_g, e_p, nid_size, heads, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(DPSTGC_block, self).__init__()    #in_channels = 1，time_strides = num_of_hours = 1
        self.DEVICE = DEVICE
        self.tcl = temporal_conv_layer(kt, in_channels, nb_tempo_filter , act="GLU")
        dim = 16
        self.gl = GraphLearn( DEVICE,num_of_vertices, dim)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, nb_tempo_filter, nb_chev_filter)
        self.nid_size = nid_size
        self.w = nn.Parameter(torch.ones(2))
        self.GAT = GAT(DEVICE, nb_tempo_filter, nb_chev_filter, num_of_vertices, e_p, self.nid_size, heads, num_of_timesteps)
        self.tcl_last1 = temporal_conv_layer(kt, nb_chev_filter, nb_time_filter, act="GLU")
        self.tcl_last2 = temporal_conv_layer(5, nb_time_filter, last_filter, act="GLU")
        #self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, last_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(last_filter)  #需要将channel放到最后一个维度上，时间维度上归一化

    def forward(self, x, norm_Adj_matrix, static_delay_matrix, _from, _to):
        '''
        :param static_delay_matrix:
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # TAt
        x_TAt = self.tcl(x.permute(0, 2, 1, 3))  # (b,f,n,t)
        x_TAt = x_TAt.permute(0, 2, 1, 3)  # 得到（b,n,t,f）
        # SAt
        static_delay_matrix = static_delay_matrix.to(self.DEVICE)
        spatial_At = self.gl(norm_Adj_matrix)
        w1 = torch.exp(self.w[0])/torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x_TAt, spatial_At)  # (b,N,F,T)
        spatial_gcn = spatial_gcn.permute(0, 1, 3, 2)
        spatial_GAT = self.GAT(x_TAt, static_delay_matrix)
        spatial_GAT = spatial_GAT.permute(0, 1, 3, 2)
        outputs = spatial_GAT * w1 + spatial_gcn * w2#(b,n,t,f_out)
        outputs = outputs.reshape(batch_size, num_of_vertices, num_of_timesteps, -1).permute(0, 1, 3, 2)
        time_conv_output = self.tcl_last1(outputs.permute(0, 2, 1, 3))
        # print(outputs.permute(0, 2, 1, 3).shape, time_conv_output.shape)
        time_conv_output = self.tcl_last2(time_conv_output)
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        return x_residual

class DPSTGC_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels,kt, K, nb_chev_filter,nb_tempo_filter, e_g, e_p, nid, heads, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(DPSTGC_submodule, self).__init__()
        in_channels,nb_tempo_filter,nb_chev_filter,nb_time_filter,last_filter = 1,32,64,32,32
        self.BlockList = nn.ModuleList([DPSTGC_block(DEVICE, in_channels,kt, K, nb_chev_filter,nb_tempo_filter,last_filter, e_g, e_p, nid, heads, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input)])
        in_channels, nb_tempo_filter, nb_chev_filter, nb_time_filter, last_filter = 32, 64, 128, 64, 64
        self.BlockList.extend([DPSTGC_block(DEVICE, in_channels, kt, K, nb_chev_filter, nb_tempo_filter,
                                                     last_filter, e_g, e_p, nid, heads, nb_time_filter, time_strides,
                                                     cheb_polynomials, num_of_vertices, len_input)])
        #每个子组件包括多个时空模块，这里是先加入一个，然后用extend在列表末尾追加nb_block-1个，也就是总共nb_block个
        in_channels, nb_tempo_filter, nb_chev_filter, nb_time_filter,last_filter = 64, 64, 128, 64,64
        self.BlockList.extend([DPSTGC_block(DEVICE, in_channels, kt, K, nb_chev_filter,nb_tempo_filter,last_filter, e_g, e_p, nid, heads, nb_time_filter, 1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, last_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x, norm_Adj_matrix, static_delay_matrix, _from, _to):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        #将输入进行若干个时空模块处理
        for block in self.BlockList:
            x = block(x, norm_Adj_matrix, static_delay_matrix, _from, _to)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output
def make_model(DEVICE, nb_block, in_channels, kt, K, nb_chev_filter,nb_tempo_filter, e_g, e_p, nid, heads, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices):
    '''

    :param e_g:
    :param e_p:
    :param nid:
    :param heads:
    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]#此处是真正调用切比雪夫多项式的，传入参数拉普拉斯矩阵和K阶
    model = DPSTGC_submodule(DEVICE, nb_block, in_channels,kt, K, nb_chev_filter,nb_tempo_filter, e_g, e_p, nid, heads, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model