# -*- coding: utf-8 -*-
"""
torch_geometric.data_Data_Batch_test.py
知行合一and至于至善 已于 2022-03-26
https://blog.csdn.net/qq_41800917/article/details/120444534
Created on Wed Jul 13 16:23:22 2022

@author: mqm
"""

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])
x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])
x_t = torch.randn(4, 16)  # 4 nodes.

edge_index_3 = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x_3 = torch.randn(4, 16)

data1= Data(x=x_s,edge_index=edge_index_s)
data2= Data(x=x_t,edge_index=edge_index_t)
data3= Data(x=x_3,edge_index=edge_index_3)
#上面是构建3张Data图对象
# * `Batch(Data)` in case `Data` objects are batched together
#* `Batch(HeteroData)` in case `HeteroData` objects are batched together

data_list = [data1, data2,data3]


loader = Batch.from_data_list(data_list)#调用该函数data_list里的data1、data2、data3 三张图形成一张大图，也就是batch
print('data_list:\n',data_list)
#data_list: [Data(edge_index=[2, 4], x=[5, 16]), Data(edge_index=[2, 3], x=[4, 16]), Data(edge_index=[2, 4], x=[4, 16])]
print('batch:',loader.batch)
#batch: tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
print('loader:',loader)
#loader: Batch(batch=[13], edge_index=[2, 11], x=[13, 16])
print('loader.edge_index:\n',loader.edge_index) #batch的边的元组
#loader.edge_index:
#tensor([[ 0,  0,  0,  0,  5,  5,  5,  9, 10, 10, 11],
#        [ 1,  2,  3,  4,  6,  7,  8, 10,  9, 11, 10]])

print('loader.num_graphs:',loader.num_graphs)#该batch的图的个数，这里是3个
#loader.num_graphs: 3

Batch=Batch.to_data_list(loader)#大图Batch变回成3张小图
print(Batch)
#[Data(edge_index=[2, 4], x=[5, 16]), Data(edge_index=[2, 3], x=[4, 16]), Data(edge_index=[2, 4], x=[4, 16])]

print('data1:',data1)

# =============================================================================
#  值得注意的是，在官方文档中并没有提及如何将自己的Data实例转换成DataLoader，
# 经搜索后发现了下面的方法，from torch_geometric.data.Data
# from torch_geometric.dataloader.DataLoader
# # data_list = [data1, data2, data3, ...]
# loader = DataLoader(data_list, batch_size=32, shuffle=True)
# 直接把存有data的列表传进DataLoader就可以了，不需要自己创建dataset。
# PyG包含它自己的torch_geometric.loader.DataLo
# 【PyG】简介 - 图神经网络
# https://blog.csdn.net/qq_40344307/article/details/122160733
# https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
# =============================================================================

from torch_geometric.loader import DataLoader
loader1 = DataLoader(data_list, batch_size=3, shuffle=False)
# data_loader=loader(0)  #取第一个图
print((loader1))
print(len(loader1))
