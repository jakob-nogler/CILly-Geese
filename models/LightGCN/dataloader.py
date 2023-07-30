import numpy as np
import scipy
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_data (df):
    indexes = df['Id'].str.extract(r'r(\d+)_c(\d+)').values.astype(int) - 1
    users, movies = (np.squeeze(x) for x in np.split(indexes, 2, axis=-1))
    return users, movies, df['Prediction'].values.astype(np.float32)


def get_dataloader (df, batch_size, with_ratings=True):
    users, movies, predictions = (torch.tensor(x, device=device) for x in parse_data(df))
    if with_ratings:
        dataset = TensorDataset(users, movies, predictions)
    else:
        dataset = TensorDataset(users, movies)
    return DataLoader(dataset, batch_size=batch_size)


def make_training_mask (dataloader, num_users, num_items):
    mask = np.full((num_users, num_items), 0)
    train_sp = np.full((num_users, num_items), 0)
    for batch in dataloader:
        users, movies, ratings = (list(x) for x in batch)
        for user, movie, rating in zip(users, movies, ratings):
            mask[user.item()][movie.item()] = 1
            train_sp[user.item()][movie.item()] = rating
    return scipy.sparse.csr_matrix(mask), \
           scipy.sparse.csr_matrix(train_sp)


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def get_sparse_graph(training_mask, num_users, num_items):
    '''Uses the matrix form build of the graph proposed in the paper'''
    adj_mat_size = num_users + num_items
    adj_mat = sp.dok_matrix((adj_mat_size, adj_mat_size), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = training_mask
    adj_mat[:num_users, num_users:] = R
    adj_mat[num_users:, :num_users] = R.T

    # degree matrix, counting non-zero entries in each row
    degree_vector = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(degree_vector, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_inv = sp.diags(d_inv)

    # symmetrically normalized matrix
    norm_adj = d_inv.dot(adj_mat).dot(d_inv).tocsr()

    graph = _convert_sp_mat_to_sp_tensor(norm_adj)
    return graph
