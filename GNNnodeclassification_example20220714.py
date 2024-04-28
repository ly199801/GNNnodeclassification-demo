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
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import torch
from torch_geometric.data import InMemoryDataset, Data

# build a graph
G = nx.Graph()
edgelist = [(0, 1), (0, 2), (1, 3)]  # note that the order of edges
G.add_edges_from(edgelist)

# x is the node feature,the nim. of rows of x is the num of nodes in G
x = torch.eye(G.number_of_nodes(), dtype=torch.float)
adj = nx.to_scipy_sparse_array(G).tocoo()
# adj = nx.to_scipy_sparse_matrix(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)

# Compute communities.
partition = community_louvain.best_partition(G)
y = torch.tensor([partition[i] for i in range(G.number_of_nodes())])

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
# print(len(data)) #len()：返回数据集中的样本的数量。
# debug1=1

def train():
    optimizer.zero_grad()
    out = model()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []

    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred .eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs


for epoch in range(1, 11):
    train()

    # print("ACC: % 0.2f, % 0.2f" % (epoch, test()))
    print("ACC:", epoch, test(), sep=',')
    # log = 'Epoch:{:03d},Train:{:.4f},Test:{:.4f}'
    # print(log.format(epoch, test()))
