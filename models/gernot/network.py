import torch
from torch import nn


class Network(nn.Module):
    '''
        Implementation of the LightGCN model
        https://github.com/gusye1234/LightGCN-PyTorch
    '''

    def __init__(self, num_users: int, num_items: int, sparse_graph, norm_means, norm_stds, user_dataloader, movie_dataloader, norm_mode='users', embedding_dim=16, hidden_dimension=64, n_layers=3, use_dropout=True, keep_probability=1/3, activation="relu"):
        super().__init__()
        self.n_users = num_users
        self.m_items = num_items
        self.use_dropout = use_dropout
        self.keep_probability = keep_probability

        self.user_dataloader = user_dataloader
        self.movie_dataloader = movie_dataloader

        self.n_layers = n_layers

        self.user_encoder = MLP(
            num_items, hidden_dimension=hidden_dimension, output_dimension=embedding_dim, activation=activation)

        self.movie_encoder = MLP(
            num_users, hidden_dimension=hidden_dimension, output_dimension=embedding_dim, activation=activation)

        self.final_mlp = MLP(input_dimension=2*embedding_dim,
                             hidden_dimension=hidden_dimension, output_dimension=1, activation=activation)

        self.graph = sparse_graph.coalesce()

        self.norm_means = torch.FloatTensor(norm_means)
        self.norm_stds = torch.FloatTensor(norm_stds)
        self.norm_mode = norm_mode

    def _apply(self, fn):
        # overridden to make the 'to' function work
        super()._apply(fn)
        self.graph = fn(self.graph)
        self.norm_means = fn(self.norm_means)
        self.norm_stds = fn(self.norm_stds)
        return self

    def __dropout(self, x, keep_probability):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_probability
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_probability
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def get_embeddings(self):
        '''propagate methods for lightGCN'''
        all_embs = []
        for users in self.user_dataloader:
            all_embs.append(self.user_encoder(users))

        for movies in self.movie_dataloader:
            all_embs.append(self.movie_encoder(movies))

        all_emb = torch.cat(all_embs)
        embs = [all_emb]

        if self.use_dropout and self.training:
            g_dropped = self.__dropout(self.graph, self.keep_probability)
        else:
            g_dropped = self.graph

        for _ in range(self.n_layers):
            # Note: it's suggested not to split user-item matrix for performance reasons
            all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)

        # simplified model using the same weight 1/(l + 1) for each layer
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return torch.split(light_out, [self.n_users, self.m_items])

    def forward(self, users, items):
        '''Uses the inner product between users and items embeddings to compute the prediction'''
        all_users, all_items = self.get_embeddings()

        users_emb = all_users[users]
        items_emb = all_items[items]
        gamma = self.final_mlp(
            torch.cat([users_emb, items_emb], dim=1)).reshape(-1)

        norm_indexes = items if self.norm_mode == 'items' else users
        means = self.norm_means[norm_indexes]
        std = self.norm_stds[norm_indexes]

        return (gamma * std) + means


class MLP(nn.Module):

    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, activation: str = "ReLu"):
        super().__init__()
        self.input_layer = nn.Linear(
            in_features=input_dimension, out_features=hidden_dimension)
        self.output_layer = nn.Linear(
            in_features=hidden_dimension, out_features=output_dimension)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.output_layer(x))
        return x


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        return nn.SELU()
