from torch_geometric.datasets import Planetoid, Amazon, CitationFull, AttributedGraphDataset
import torch
from Noise_about import noisify_with_P
import numpy as np
import random
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops, add_self_loops
from copy import deepcopy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(name, split='per_class', path='./data', seed=0, hop=2):
    setup_seed(seed)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=path, name=name)
    elif name in ['dblp']:
        dataset = CitationFull(root=path, name=name)
    else:
        raise ValueError("Invalid dataset name.")

    data = dataset[0]
    
    all_num = len(data.y)
    
    if name in ['dblp']:
        remaining_indices = list(range(all_num))
        if split == 'percent':
            # percent
            train_size = int(0.01 * len(data.y))
            val_size = int(0.15 * len(data.y))
            test_size = len(data.y) - train_size - val_size
            train_indices = np.random.choice(remaining_indices, train_size, replace=False)
        elif split == 'per_class':
            # sample per class
            train_examples_per_class = 20
            val_size = 500
            test_size = 1000
            train_indices = sample_per_class(data.y, train_examples_per_class)
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)
        forbidden_indices = np.concatenate((train_indices, val_indices))
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    
        train_mask = torch.zeros(all_num, dtype=torch.bool)
        val_mask = torch.zeros(all_num, dtype=torch.bool)
        test_mask = torch.zeros(all_num, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
    add_distant_neighbors(data, hop)

    print(f"""----Data statistics------'
        Name: {name}
        #Nodes {len(data.y)}
        #Edges {data.num_edges // 2}
        #Classes {dataset.num_classes}
        #Train:val:test = {data.train_mask.sum()} : {data.val_mask.sum()} : {data.test_mask.sum()}""")
    

    return dataset, data


def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def add_noise(data, num_classes, noise_rate = 0.4, noise_type='uniform', random_state=0):
    y_hat, _, real_noise = noisify_with_P(data.y, num_classes, noise_rate, random_state=random_state, noise_type=noise_type)
    y_hat = torch.tensor(y_hat)
    actual_noise = (data.y != y_hat)
    noise_indices = torch.where(actual_noise)[0]
    ori_indices = torch.where(data.train_mask)[0]
    noise_indices = list(set(noise_indices.tolist()) & set(ori_indices.tolist()))
    clean_indices = list(set(ori_indices.tolist()) - set(noise_indices))

    data.y[data.train_mask] = y_hat[data.train_mask]
    data.y[data.val_mask] = y_hat[data.val_mask]

    print(f"""----Noise Process------'
        Type: {noise_type}
        Target noise rate: {noise_rate * 100}%
        State: {random_state}
        Actual noise: {real_noise * 100 : .2f}%
        noise: clean = {len(noise_indices)} : {len(clean_indices)}""")

    return data, ori_indices, clean_indices, noise_indices

def add_distant_neighbors(data, hops):
    """Add multi_edge_index attribute to data which includes the edges of 2,3,... hops neighbors."""
    assert hops > 1
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index,
                                   num_nodes=data.x.size(0))
    one_hop_set = set([tuple(x) for x in edge_index.transpose(0, 1).tolist()])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col)
    multi_adj = adj
    for _ in range(hops - 1):
        multi_adj = multi_adj @ adj
    row, col, _ = multi_adj.coo()
    multi_edge_index = torch.stack([row, col], dim=0)
    multi_hop_set = set([tuple(x) for x in multi_edge_index.transpose(0, 1).tolist()])
    multi_hop_set = multi_hop_set - one_hop_set
    multi_edge_index = torch.LongTensor(list(multi_hop_set)).transpose(0, 1)
    data.multi_edge_index = multi_edge_index
    return