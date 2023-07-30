from ..model import Model
from .neural_net import NeuralNet

import torch as T
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from sklearn.metrics import mean_squared_error

import random
device = T.device("cuda" if T.cuda.is_available() else "cpu")


'''
HYPERPARAMETERS
'seed' : the seed that we are using
'hidden_dimension' = 128
'encoded_dimension' = 24
'learning_rate' = 1e-3
'weight_decay' = 1e-4
'batch_size' = 256
'num_epochs' = 200
'''

number_of_users, number_of_movies = (10000, 1000)


class VAE(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        np.random.seed(hyperparameters['seed'])
        T.manual_seed(hyperparameters['seed'])
        random.seed(hyperparameters['seed'])

        self.preprocess_data()

        self.model = NeuralNet(
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            hidden_dimension=hyperparameters["hidden_dimension"],
            encoded_dimension=hyperparameters["encoded_dimension"],
            num_hidden_encoder=hyperparameters["num_hidden_encoder"],
            num_hidden_decoder=hyperparameters["num_hidden_decoder"],
            drop_probability=hyperparameters["drop_probability"]
        ).to(device)

        if hyperparameters["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=hyperparameters['learning_rate'],
                                        weight_decay=hyperparameters['weight_decay'])
        elif hyperparameters["optimizer"] == "momentum":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=hyperparameters['learning_rate'],
                                       weight_decay=hyperparameters['weight_decay'],
                                       momentum=hyperparameters["momentum"])

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=20,
            factor=0.5,
        )

        self.dataloader = DataLoader(
            TensorDataset(self.input_matrix,
                          self.output_matrix, self.output_mask),
            batch_size=hyperparameters['batch_size'],
        )

        self.epoch = -1
        self.last_tested_epoch = -1
        self.last_test_result = -1

    # do one epoch of training
    def train(self):
        self.epoch = self.epoch + 1

        loss_sum = 0
        loss_count = 0
        self.model.train()
        self.model.set_training()
        for data_batch, review_batch, mask_batch in self.dataloader:

            self.optimizer.zero_grad()

            reconstructed, mu, logvar = self.model(data_batch)

            loss = self.loss_function(
                review_batch, reconstructed, mask_batch, mu, logvar)
            loss_sum += loss.item()
            loss_count += 1
            loss.backward()

            self.optimizer.step()

        self.model.eval()
        self.scheduler.step(self.test())

        return loss.item()

    # predicts on the test set and return the score
    def test(self) -> float:
        if self.epoch == self.last_tested_epoch:
            return self.last_test_result
        reconstructed_matrix = self.reconstruct_whole_matrix()

        predictions = np.zeros(len(self.t_users))

        for i, (user, movie) in enumerate(zip(self.t_users, self.t_movies)):
            predictions[i] = reconstructed_matrix[user][movie]
        score = self.get_score(predictions, self.t_predictions)

        self.last_test_result = self.epoch
        self.last_test_result = score
        return score

    # predicts the reviews
    def predict(self, to_predict):
        reconstructed_matrix = self.reconstruct_whole_matrix()

        users, movies = [np.squeeze(arr) for arr in np.split(
            to_predict.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        preds = []

        for user, movie in zip(users, movies):
            preds.append(reconstructed_matrix[user][movie])

        preds = np.array(preds)

        to_predict["Prediction"] = preds
        return to_predict

    def preprocess_data(self):

        # test data
        self.t_users, self.t_movies, self.t_predictions = self.extract_data(
            self.test_set)

        # train data
        users, movies, predictions = self.extract_data(self.train_set)

        # save data as matrix
        data = np.full((number_of_users, number_of_movies),
                       np.mean(self.train_set.Prediction.values))
        mask = np.zeros((number_of_users, number_of_movies))

        for user, movie, pred in zip(users, movies, predictions):
            data[user][movie] = pred
            mask[user][movie] = 1

        if self.hyperparameters["transpose"]:
            self.number_rows = number_of_movies
            self.number_columns = number_of_users
            data = np.transpose(data)
            mask = np.transpose(mask)
        else:
            self.number_rows = number_of_users
            self.number_columns = number_of_movies

        self.output_matrix = np.zeros(
            (self.number_rows, self.number_columns), dtype=np.float32)
        self.output_mask = np.ones(
            (self.number_rows, self.number_columns), dtype=np.int32)

        self.output_dimension = self.number_columns

        if self.hyperparameters["one_hot_reviews"]:
            self.input_matrix = np.zeros(
                (self.number_rows, self.number_columns * 6), dtype=np.float32)
            self.input_dimension = self.number_columns * 6
            for row in range(self.number_rows):
                for col in range(self.number_columns):
                    if not mask[row][col]:
                        self.input_matrix[row, 6 * col + 5] = 1
                        self.output_mask[row, col] = 0
                    else:
                        self.input_matrix[row, 6*col +
                                          int(data[row][col]) - 1] = 1
                        self.output_matrix[row, col] = data[row][col]
        else:
            self.input_matrix = np.zeros(
                (self.number_rows, self.number_columns * 2), dtype=np.float32)
            self.input_dimension = self.number_columns * 2
            for row in range(self.number_rows):
                for col in range(self.number_columns):
                    if not mask[row][col]:
                        self.input_matrix[row, 2 * col + 1] = 1
                        self.output_mask[row, col] = 0
                    else:
                        self.input_matrix[row, 2*col] = data[row][col]
                        self.output_matrix[row, col] = data[row][col]

        self.input_matrix = T.tensor(self.input_matrix, device=device).float()
        self.output_matrix = T.tensor(
            self.output_matrix, device=device).float()
        self.output_mask = T.tensor(self.output_mask, device=device)

    def extract_data(self, data):
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        predictions = data.Prediction.values
        return users, movies, predictions

    def loss_function(self, original, reconstructed, mask, mu, logvar):
        ll = T.mean(mask * (original - reconstructed) ** 2)
        kld = -0.05 * \
            T.mean(T.sum(1 + logvar - T.pow(mu, 2) - T.exp(logvar), axis=1))
        return ll + kld

    def reconstruct_whole_matrix(self,):
        data_reconstructed = np.zeros((self.number_rows, self.number_columns))
        self.model.set_inference()
        with T.no_grad():
            for i in range(0, self.number_rows, self.hyperparameters['batch_size']):
                upper_bound = min(
                    i + self.hyperparameters['batch_size'], self.number_rows)
                predictions, _, _ = self.model(
                    self.input_matrix[i:upper_bound])

                reconstructed = predictions
                data_reconstructed[i:upper_bound] = reconstructed.detach(
                ).cpu().numpy()

        if self.hyperparameters["transpose"]:
            data_reconstructed = np.transpose(data_reconstructed)

        return data_reconstructed

    def rmse(self, x, y): return math.sqrt(mean_squared_error(x, y))

    def get_score(self, predictions, target_values):
        return self.rmse(predictions, target_values)

    # saves the model under the specified path
    def save(self, path: str):
        T.save(self.model.state_dict(), path)

    # loads the model from the specified path
    def load(self, path: str):
        self.model.load_state_dict(T.load(path, map_location=device))
