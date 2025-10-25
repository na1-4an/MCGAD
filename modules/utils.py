import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'  # 또는 고정값
import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import random
import shutil
import scipy.io as sio
import scipy.sparse as sp
from tqdm import tqdm
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    if 'Label' in data:
        labels = np.squeeze(np.array(data['Label'],dtype=np.int64) - 1) 
    if 'Class' in data:
        labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1) 
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels

def load_dataset(dataset):
    print('Dataset: {}'.format(dataset), flush=True)
    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)
    features, _ = preprocess_features(features)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)
    return g, features, ano_label, str_ano_label, attr_ano_label, edge_index


def set_random_seeds(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 

def rescale(x):
    return (x + 1) / 2


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_lap(edge_index, n):
    print(n)
    edge_indext = edge_index.T
    adjacency = csr_matrix((np.ones(edge_indext.shape[0]), (edge_indext[:, 0], edge_indext[:, 1])), shape=(n, n))
    nadj = csgraph.laplacian(adjacency, normed=True)
    return sparse_mx_to_torch_sparse_tensor(nadj)
