from ..model import Model
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import math
import pandas as pd


from .neural_net import NeuralNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFDA(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        self.epoch = -1
        self.batch_size = hyperparameters['batch_size']
        self.use_all_interactions = hyperparameters.get(
            "all_interactions", True)

        self.preprocess_data()

        self.model = NeuralNet(
            user_input_size=self.number_of_movies,
            movie_input_size=self.number_of_users,
            hidden_dimension_user=self.hyperparameters["hidden_dimension_user"],
            encoded_dimension_user=self.hyperparameters["encoded_dimension_user"],
            hidden_dimension_movie=self.hyperparameters["hidden_dimension_movie"],
            encoded_dimension_movie=self.hyperparameters["encoded_dimension_movie"],
            hidden_dimension_predictor=self.hyperparameters["hidden_dimension_predictor"],
            drop_probability=self.hyperparameters["drop_probability"],
            activation=self.hyperparameters["activation"]
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=hyperparameters['learning_rate'],
                                    weight_decay=hyperparameters["weight_decay"])

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=self.hyperparameters['patience'],
            factor=0.5
        )

        self.extension = int(
            self.hyperparameters['extension_factor'] * self.batch_size)

    def train(self):
        self.epoch += 1

        loss_sum = 0

        self.model.train()

        loss_count = 0

        for user_inx, movie_inx in self.train_dataloader:

            user_inx_1 = torch.randint(
                0, self.number_of_users, (self.extension, ), device=device)
            movie_inx_1 = torch.randint(
                0, self.number_of_movies, (self.extension, ), device=device)
            user_inx = torch.cat((user_inx, user_inx_1))
            movie_inx = torch.cat((movie_inx, movie_inx_1))
            self.optimizer.zero_grad()
            users = self.matrix[user_inx]
            movies = self.matrixT[movie_inx]

            user_mask = self.mask[user_inx]
            movie_mask = self.mask.transpose(0, 1)[movie_inx]

            flat_inx = user_inx * self.number_of_movies + movie_inx

            reviews = self.matrix.reshape(-1)[flat_inx]
            mask = self.mask.reshape(-1)[flat_inx]

            if self.use_all_interactions:
                true_interactions = self.true_interactions.reshape(-1)[
                    flat_inx]
            else:
                true_interactions = mask

            review_prediction, interaction_prediction, user_prediction, movie_prediction = self.model(
                users, movies)
            loss = 0
            pred_loss = self.predictor_loss(
                review_prediction, reviews, interaction_prediction, mask, true_interactions)
            loss += pred_loss

            if self.hyperparameters['train_embeddings']:
                user_loss = self.encoder_loss(
                    user_prediction, users, user_mask)
                loss += user_loss
                movie_loss = self.encoder_loss(
                    movie_prediction, movies, movie_mask)
                loss += movie_loss

            loss_sum += loss.cpu().item()
            loss_count += 1

            loss.backward()
            self.optimizer.step()

        # print(f"Loss | pred {pred_loss} | user {user_loss} | movie {movie_loss}")
        self.model.eval()
        self.scheduler.step(self.test())

        return loss_sum / loss_count

    def test(self):
        with torch.no_grad():
            all_predictions = []
            for users, movies in self.test_dataloader:
                users = self.matrix[users]
                movies = self.matrixT[movies]
                predictions, _, _, _ = self.model(users, movies)
                all_predictions.append(predictions)
        all_predictions = torch.cat(all_predictions)
        score = self.get_score(
            all_predictions.cpu(), self.test_predictions.cpu()
        )

        return score

    def predict(self, to_predict):
        users, movies, _ = self.extract_data(to_predict)
        users = torch.tensor(users, device=device)
        movies = torch.tensor(movies, device=device)

        dataloader = DataLoader(TensorDataset(
            users, movies), batch_size=self.batch_size)
        with torch.no_grad():
            all_predictions = []
            for users, movies in dataloader:
                users = self.matrix[users]
                movies = self.matrixT[movies]
                predictions, _, _, _ = self.model(users, movies)
                all_predictions.append(predictions)
        all_predictions = torch.cat(all_predictions)
        to_predict["Prediction"] = all_predictions.cpu().numpy()
        return to_predict

    def encoder_loss(self, prediction, real, mask):
        return torch.mean(mask * (prediction - real)**2)

    def predictor_loss(self, review_predictions, reviews, interaction_predictions, mask, true_interactions):
        return torch.mean(mask * (review_predictions - reviews) ** 2 + (true_interactions - interaction_predictions)**2)

    def get_score(self, predictions, target_values):
        def rmse(x, y): return math.sqrt(mean_squared_error(x, y))
        return rmse(predictions, target_values)

    def preprocess_data(self):

        test_users, test_movies, test_predictions = self.extract_data(
            self.test_set)
        users, movies, predictions = self.extract_data(self.train_set)

        self.number_of_users = max(np.max(users), np.max(test_users)) + 1
        self.number_of_movies = max(np.max(test_movies), np.max(movies)) + 1

        # matrix with all interactions
        true_interactions = np.zeros(
            (self.number_of_users, self.number_of_movies), dtype=np.int32)

        user_movie_pairs = pd.read_csv("./data/user_movie_pairs.csv")

        for user, movie in zip(user_movie_pairs.user_id, user_movie_pairs.movie_id):
            true_interactions[user - 1][movie - 1] = 1

        test_users = torch.tensor(test_users, device=device)
        test_movies = torch.tensor(test_movies, device=device)
        self.test_predictions = torch.tensor(test_predictions, device=device)

        self.test_dataloader = DataLoader(TensorDataset(
            test_users, test_movies), batch_size=self.batch_size)

        matrix = np.zeros(
            (self.number_of_users, self.number_of_movies), dtype=np.float32)
        mask = np.zeros(
            (self.number_of_users, self.number_of_movies), dtype=np.int32)

        for user, movie, prediction in zip(users, movies, predictions):
            matrix[user][movie] = prediction
            mask[user][movie] = 1

        self.matrix = torch.tensor(matrix, device=device)
        self.matrixT = torch.tensor(matrix.transpose(), device=device)
        self.mask = torch.tensor(mask, device=device)
        self.true_interactions = torch.tensor(true_interactions, device=device)

        if False:  # self.use_all_interactions:
            train_users = torch.tensor(
                user_movie_pairs.user_id - 1, dtype=torch.int32, device=device)
            train_movies = torch.tensor(
                user_movie_pairs.movie_id - 1, dtype=torch.int32, device=device)
        else:
            train_users = torch.tensor(users, device=device)
            train_movies = torch.tensor(movies, device=device)
        self.train_dataloader = DataLoader(TensorDataset(
            train_users, train_movies), batch_size=self.batch_size, shuffle=True)

    def extract_data(self, data):
        """Reads the data from the pd.DataFrame in the kaggle format
        """
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        predictions = data.Prediction.values
        return users, movies, predictions

    # saves the model under the specified path
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    # loads the model from the specified path
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=device))
