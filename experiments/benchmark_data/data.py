import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.utils import add_remaining_self_loops, to_undirected, coalesce, softmax
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor
from torch_geometric.data import Data

# ============== Data Loading ==============
def get_data(name, split=0):
    path = './data/' + name

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=path, name=name)
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['cornell', 'texas']:
        dataset = WebKB(root=path, name=name)
    elif name == 'film':
        dataset = Actor(root=path)

    data = dataset[0]
    data = preprocess_graph(data)

    if name in ['cora', 'citeseer', 'pubmed']:
        return data

    if name in ['chameleon', 'squirrel']:
        splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
    elif name in ['cornell', 'texas']:
        splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
    elif name == 'film':
        splits_file = np.load(f'{path}/raw/{name}_split_0.6_0.2_{split}.npz')

    data.train_mask = torch.tensor(splits_file['train_mask'], dtype=torch.bool)
    data.val_mask = torch.tensor(splits_file['val_mask'], dtype=torch.bool)
    data.test_mask = torch.tensor(splits_file['test_mask'], dtype=torch.bool)

    return data


def create_random_splits(data, num_splits=10, train_ratio=0.6, val_ratio=0.2, seed=42):
    n = data.num_nodes
    splits = []
    rng = np.random.RandomState(seed)

    for _ in range(num_splits):
        idx = rng.permutation(n)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[idx[:train_end]] = True
        val_mask[idx[train_end:val_end]] = True
        test_mask[idx[val_end:]] = True

        splits.append((train_mask, val_mask, test_mask))

    return splits
