# from dataloaders.igmc import SparseRowIndexer, SparseColIndexer

import torch
import numpy as np
from torch_geometric.data import Data
from scipy import sparse
import random
import scipy.sparse as sp
import time


def extract_subgraph(SRI, SCI, u, v, y, max_nodes):
    u, v, r, node_labels = extract_neighbor(SRI, SCI, u, v, max_nodes)
    r -= 1
    data = process_input(u=u,
                         v=v,
                         r=r,
                         node_labels=node_labels,
                         max_node_label=3,
                         y=y
                         )
    return data


def extract_neighbor(SRI, SCI, u, v, max_nodes):
    u_nodes = [int(u)]
    v_nodes = [int(v)]

    u_neighbor = neighbor(SRI, u_nodes)
    v_neighbor = neighbor(SCI, v_nodes)

    if len(u_neighbor) > max_nodes:
        u_neighbor = list(random.sample(u_neighbor, max_nodes))
    else:
        u_neighbor = list(u_neighbor)

    if len(v_neighbor) > max_nodes:
        v_neighbor = list(random.sample(v_neighbor, max_nodes))
    else:
        v_neighbor = list(v_neighbor)

    u_nodes = u_nodes + v_neighbor
    v_nodes = v_nodes + u_neighbor

    subgraph = SRI[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0  # delete target link
    u, v, r = sp.find(subgraph)

    node_labels = [0] + [2] * (len(u_nodes)-1) + [1] + [3] * (len(v_nodes)-1)

    return u, v, r, node_labels


def neighbor(adj_matrix, node):
    return set(adj_matrix[list(node)].indices)


def process_input(u, v, r, node_labels, max_node_label, y):
    u, v, r = torch.LongTensor(u), torch.LongTensor(v), torch.LongTensor(r)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    # print("r: ", r)
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    y = torch.FloatTensor([y])
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    # print(data)
    return data


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def unused_neighbor(adj_matrix, sparse_matrix, u, v, max_nodes):
    """
    This was my naive attempt to extract the subgraph.
    Its high time complexity renders it unusable T_T
    """
    start = time.time()
    u_neighbor = [[v_] for u_, v_, _ in sparse_matrix if u_ == u and v_ != v]
    v_neighbor = [[u_] for u_, v_, _ in sparse_matrix if v_ == v and u_ != u]
    end = time.time()
    print("First half of neighbor took: ", end - start)

    if len(u_neighbor) > max_nodes:
        u_neighbor = np.array(random.sample(u_neighbor, max_nodes)).reshape(-1).astype(np.int32)
    else:
        u_neighbor = np.array(u_neighbor).reshape(-1).astype(np.int32)

    if len(v_neighbor) > max_nodes:
        v_neighbor = np.array(random.sample(v_neighbor, max_nodes)).reshape(-1).astype(np.int32)
    else:
        v_neighbor = np.array(v_neighbor).reshape(-1).astype(np.int32)


    # This is just sad programming
    # start = time.time()
    temp = np.zeros((v_neighbor.shape[0], u_neighbor.shape[0]))
    for i in range(v_neighbor.shape[0]):
        for j in range(u_neighbor.shape[0]):
            temp[i, j] = adj_matrix[v_neighbor[i], u_neighbor[j]]
    neighbor_list = sp.csr_matrix(temp)
    neighbor_list = np.array(sp.find(neighbor_list))
    # end = time.time()
    # print("neighbor_list took: ", end-start)

    node_labels = np.array([0] + [2]*u_neighbor.shape[0] + [1] + [3]*v_neighbor.shape[0])
    neighbor_list[2] -= 1

    return neighbor_list, node_labels
