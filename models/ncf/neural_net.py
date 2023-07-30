import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, number_of_users: int, number_of_movies: int, users_embedding_size: int, movies_embedding_size: int, num_layers: int, hidden_dimension: int, drop_probability: float = 0.0):
        super().__init__()

        self.embedding_layer_users = nn.Embedding(
            number_of_users, users_embedding_size)
        self.embedding_layer_movies = nn.Embedding(
            number_of_movies, movies_embedding_size)

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(in_features=users_embedding_size +
                      movies_embedding_size, out_features=hidden_dimension)
        )
        for i in range(num_layers-1):
            self.hidden_layers.append(
                nn.Linear(in_features=hidden_dimension // (2**i),
                          out_features=hidden_dimension // (2**(i+1)))
            )
        self.hidden_layers.append(
            nn.Linear(in_features=hidden_dimension //
                      (2**(num_layers - 1)), out_features=1)
        )
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(drop_probability)

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        x = concat
        for layer in self.hidden_layers:
            x = self.dropout(x)
            x = layer(x)
            x = self.activation(x)
        return torch.squeeze(x)
