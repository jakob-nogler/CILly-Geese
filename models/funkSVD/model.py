from funk_svd import SVD
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
from ..model import Model


class funkSVD(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        train_set = self.format_data(train_set)
        self.test_data = self.format_data(test_set)

        self.train_data = train_set.sample(frac=0.9, random_state=6)
        self.val = train_set.drop(self.train_data.index.tolist())

        self.lr = hyperparameters["learning_rate"]
        self.reg = hyperparameters["reg_parameter"]
        self.n_epochs = hyperparameters["n_epochs"]
        self.n_factors = hyperparameters["n_factors"]
        self.early_stopping = hyperparameters.get('early_stopping', True)
        self.rec_mtx = None
        self.svd = SVD(lr=self.lr,
                       reg=self.reg,
                       n_epochs=self.n_epochs,
                       n_factors=self.n_factors,
                       early_stopping=self.early_stopping,
                       shuffle=False,
                       min_rating=1,
                       max_rating=5)
        self.svd.fit(X=self.train_data, X_val=self.val)

    def train(self) -> float:
        return 0

    # predicts on the test set and return the score
    def test(self) -> float:
        predictions = self.svd.predict(self.test_data)
        self.test_score = self.get_score(predictions, self.test_data["rating"])
        return self.test_score

    # predicts the reviews for the actual unknowns
    def predict(self, to_predict):
        predict_data = self.format_data(to_predict)
        predictions = self.svd.predict(predict_data)

        to_predict["Prediction"] = predictions
        return to_predict

    def format_data(self, data):
        # reformats into DataFrame with columns u_id, i_id, and rating
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values, 2, axis=-1)]

        predictions = data.Prediction.values
        # roundabout way of reformatting but pandas scares me
        return pd.DataFrame({"u_id": users, "i_id": movies, "rating": predictions})

    def rmse(self, x, y): return math.sqrt(mean_squared_error(x, y))

    def get_score(self, predictions, target_values):
        return self.rmse(predictions, target_values)

    # no great saving or loading options, but it's ok bc this is very quick
    # def save(self, path: str):

    # def load(self, path: str):
