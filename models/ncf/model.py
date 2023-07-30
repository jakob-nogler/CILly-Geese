import numpy as np
import math
from sklearn.metrics import mean_squared_error

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .neural_net import NN
from .. import model
'''
HYPERPARAMETERS
'seed' : the seed that we are using
'batch_size' : 128 # 1024
'movies_embedding_size' : 16
'users_embedding_size' : 32
'learning_rate' : 1e-3 # 1e-3
'weight_decay' : 1e-4
'patience' : 5

Note: num_epochs was set to ~30
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NCF(model.Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        train_users, train_movies, train_predictions = self.extract_users_items_predictions(
            self.train_set)
        test_users, test_movies, test_predictions = self.extract_users_items_predictions(
            self.test_set)

        self.train_predictions = train_predictions.astype(float)
        self.test_predictions = test_predictions.astype(float)

        # train data loader
        train_users_torch = torch.tensor(train_users, device=device)
        train_movies_torch = torch.tensor(train_movies, device=device)
        train_predictions_torch = torch.tensor(
            train_predictions, device=device)
        self.train_dataloader = DataLoader(TensorDataset(
            train_users_torch, train_movies_torch, train_predictions_torch), batch_size=hyperparameters['batch_size'])

        # test data loader
        test_users_torch = torch.tensor(test_users, device=device)
        test_movies_torch = torch.tensor(test_movies, device=device)
        self.test_dataloader = DataLoader(TensorDataset(
            test_users_torch, test_movies_torch), batch_size=hyperparameters['batch_size'])

        # set up nn
        number_of_users, number_of_movies = (10000, 1000)
        self.nn = NN(number_of_users, number_of_movies,
                     hyperparameters['users_embedding_size'], hyperparameters[
                         'movies_embedding_size'], hyperparameters['num_layers'], hyperparameters['hidden_dimension'],
                     drop_probability=hyperparameters['drop_probability']).to(device)
        self.optimizer = optim.Adam(self.nn.parameters(
        ), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=hyperparameters['patience'])

    # do one epoch of training
    def train(self):
        total_loss = 0

        for users_batch, movies_batch, target_predictions_batch in self.train_dataloader:
            self.optimizer.zero_grad()

            predictions_batch = self.nn(users_batch, movies_batch)

            loss = self.mse_loss(predictions_batch, target_predictions_batch)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        self.scheduler.step(self.test())

        return total_loss

    # predicts on the test set and return the score
    def test(self) -> float:
        with torch.no_grad():
            all_predictions = []
            for users_batch, movies_batch in self.test_dataloader:
                predictions_batch = self.nn(users_batch, movies_batch)
                all_predictions.append(predictions_batch)

        all_predictions = torch.cat(all_predictions)
        all_predictions = self.round_tensor(all_predictions)

        reconstuction_rmse = self.get_score(
            all_predictions.cpu().numpy(), self.test_predictions)
        return reconstuction_rmse

    # predicts the reviews
    def predict(self, to_predict):
        fin_test_users, fin_test_movies, fin_test_predictions = self.extract_users_items_predictions(
            to_predict)

        fin_test_users_torch = torch.tensor(fin_test_users, device=device)
        fin_test_movies_torch = torch.tensor(
            fin_test_movies, device=device)

        fin_test_dataloader = DataLoader(TensorDataset(
            fin_test_users_torch, fin_test_movies_torch), batch_size=self.hyperparameters['batch_size'])
        with torch.no_grad():
            all_predictions = []
            for fin_users_batch, fin_movies_batch in fin_test_dataloader:
                fin_predictions_batch = self.nn(
                    fin_users_batch, fin_movies_batch)
                all_predictions.append(fin_predictions_batch)

        all_predictions = torch.cat(all_predictions)
        all_predictions = self.round_tensor(all_predictions)
        to_predict["Prediction"] = all_predictions.cpu().numpy()

        return to_predict

    # saves the model under the specified path
    def save(self, path: str):
        torch.save(self.nn.state_dict(), path)

    # loads the model from the specified path
    def load(self, path: str):
        self.nn.load_state_dict(torch.load(path, map_location=device))

    def mse_loss(self, predictions, target):
        return torch.mean((predictions - target) ** 2)

    def get_score(self, predictions, target_values):
        def rmse(x, y): return math.sqrt(mean_squared_error(x, y))
        return rmse(predictions, target_values)

    def round_pred(self, val):
        return val

    def round_tensor(self, pred):
        for index in range(len(pred)):
            pred[index] = self.round_pred(pred[[index]])
        return pred

    def extract_users_items_predictions(self, data_pd):
        users, movies = \
            [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
                'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
        predictions = data_pd.Prediction.values
        return users, movies, predictions
