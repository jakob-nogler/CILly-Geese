import torch.nn as nn
import torch


class NeuralNet(nn.Module):
    def __init__(self, user_input_size: int, movie_input_size: int, hidden_dimension_user: int, encoded_dimension_user: int,
                hidden_dimension_movie : int, encoded_dimension_movie: int, hidden_dimension_predictor: int,
                drop_probability: float = 0.0, activation: str = "ReLu"):
        super().__init__()

        self.user_encoder = MLP(
            input_dimension=user_input_size,
            hidden_dimension=hidden_dimension_user,
            output_dimension=encoded_dimension_user
        )

        self.user_decoder = MLP(
            input_dimension=encoded_dimension_user,
            hidden_dimension=hidden_dimension_user,
            output_dimension=user_input_size
        )

        self.movie_encoder = MLP(
            input_dimension=movie_input_size,
            hidden_dimension=hidden_dimension_movie,
            output_dimension=encoded_dimension_movie
        )

        self.movie_decoder = MLP(
            input_dimension=encoded_dimension_movie,
            hidden_dimension=hidden_dimension_movie,
            output_dimension=movie_input_size
        )

        self.predictor = MLP(
            input_dimension=encoded_dimension_movie + encoded_dimension_user,
            hidden_dimension=hidden_dimension_predictor,
            output_dimension=2
        )


        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(drop_probability)

    def forward(self, user_array: torch.Tensor, movie_array: torch.Tensor):
        user_enc = self.dropout(self.user_encoder(user_array))
        movie_enc = self.dropout(self.movie_encoder(movie_array))


        x = torch.cat([user_enc, movie_enc], dim=1)
        pred = self.predictor(x)

        reviews = pred[:, 0]
        interactions = pred[:, 1]

        user_dec = self.user_decoder(user_enc)
        movie_dec = self.movie_decoder(movie_enc)

        return reviews, interactions, user_dec, movie_dec

class MLP(nn.Module):

    def __init__(self, input_dimension: int, hidden_dimension : int, output_dimension: int, activation: str = "ReLu"):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_dimension, out_features=hidden_dimension)
        self.output_layer = nn.Linear(in_features=hidden_dimension, out_features=output_dimension)
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
