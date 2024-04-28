# -*- coding: utf-8 -*-
"""
GNNnodeclassification_example.py
PyTorch图神经⽹络实践（⼆）⾃定义图数据
https://wenku.baidu.com/view/544541f2350cba1aa8114431b90d6c85ec3a880d.html

Created on Wed Jul 13 09:31:07 2022


"""
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import torch
from torch_geometric.data import InMemoryDataset, Data

#
N = 9
Ns = 3
efs_raw1 = pd.read_csv("effcient_solar_coord_Pattern1.csv")
efs1 = efs_raw1.T.copy()  #
efs2 = efs1.iloc[0:N, :].values

grid_raw1 = pd.read_csv("grid9_edges12.csv")  # 普通邻接表含有所有有方向的边，并且有权重
grid1 = grid_raw1.copy()  #
grid2 = grid1.iloc[:, :].values
grid3 = grid2[:, 0:2].T

G = nx.Graph()
grid4 = []
for i in range(grid3.shape[1]):
    grid0 = grid3[:, i]
    grid4 = [tuple(grid0.T)]
    # grid4=[(1,2)]
    G.add_edges_from(grid4)

debug1 = 1


# build a graph
# G = nx.Graph()
# edgelist0=grid3.tolist()
# edgelist1=()
# for i in range(len(edgelist0)):
#     edgelist1=edgelist1+tuple(edgelist0[i])
# edgelist = [(0, 1), (0, 2), (1, 3)]  # note that the order of edges
# G.add_edges_from(edgelist)

x = torch.from_numpy(efs2)  # 数组转换为张量
# x=torch.FloatTensor(x0)
# x is the node feature,the nim. of rows of x is the num of nodes in G
# x=torch.from_numpy(efs2)
# x = torch.eye(G.number_of_nodes(), dtype=torch.float)
# x = torch.cat((x, x), 1)  # concat:把几个tensor连接起来

adj = nx.to_scipy_sparse_array(G).tocoo()
# adj = nx.to_scipy_sparse_matrix(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)

# Compute communities.
# 列表转化为字典
partition_list1 = range(G.number_of_nodes())
partition_list21 = [1]*Ns
partition_list22 = [0]*(N-Ns)
partition_list2 = partition_list21+partition_list22
y = torch.LongTensor([partition_list2[i] for i in range(G.number_of_nodes())])

# partition = community_louvain.best_partition(G)
# y = torch.tensor([partition[i] for i in range(G.number_of_nodes())])

# Select a single training node for each community
# (we just use the first one).
train_mask = torch.zeros(y.size(0), dtype=torch.bool)
for i in range(int(y.max()) + 1):
    train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
    # torch.nonzero(input, *, out=None, as_tuple=False)
    # if as_tuple = False： 输出的每一行为非零元素的索引:
    # if as_tuple = True： 输出是每一个维度都有一个一维的张量

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
remaining = remaining[torch.randperm(remaining.size(0))]
data.test_mask = torch.zeros(y.size(0), dtype=torch.bool)
data.test_mask.fill_(False)
data.test_mask[remaining[:]] = True


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x=x.to(torch.float32)  #July 16 ,2022修改expected scalar type Double but found Float
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)], lr=0.01)
# Only perform weight-decay on first convolution.
# print(len(data)) #len()：返回数据集中的样本的数量。 作者：可爱滴小豆子 https://www.bilibili.com/read/cv12084371 出处：bilibili
# debug1=1


def train():
    optimizer.zero_grad()
    out = model()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    logits, accs = model(), []

    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred .eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs

num_epochs=100
loss_epochs=np.zeros(num_epochs)

for epoch in range(1, num_epochs):
    loss=train()
    loss_epochs[epoch]=loss.item()
    # print("ACC: % 0.2f, % 0.2f" % (epoch, test()))
    print("loss,ACC-train,ACC-test:", loss.item(),epoch, test(), sep=',')
    # log = 'Epoch:{:03d},Train:{:.4f},Test:{:.4f}'
    # print(log.format(epoch, test()))

plt.plot(loss_epochs)	# s为点大小
plt.show()