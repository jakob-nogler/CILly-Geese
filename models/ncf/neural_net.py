import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, number_of_users: int, number_of_movies: int, users_embedding_size: int, movies_embedding_size: int, num_layers: int, hidden_dimension: int, drop_probability: float = 0.0):
        super().__init__()
        self.embedding_layer_users = nn.Embedding(
            number_of_users, users_embedding_size)
        self.embedding_layer_movies = nn.Embedding(
            number_of_movies, movies_embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=users_embedding_size +
                      movies_embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            # maybe predict per category?
            nn.Linear(in_features=16, out_features=1),
            nn.ReLU()
        )

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))
