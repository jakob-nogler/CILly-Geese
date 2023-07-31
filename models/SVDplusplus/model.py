from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVDpp, accuracy
import math
import numpy as np
import pandas as pd
from ..model import Model


class SVDplusplus(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        self.train_data = self.format_data(train_set)
        self.test_data = self.format_data(test_set)

        self.reader = Reader(rating_scale=(1,5))
        surp_train = Dataset.load_from_df(self.train_data, self.reader)
        full_surp_trainset = surp_train.build_full_trainset()

        self.lr = hyperparameters["learning_rate"]
        self.reg = hyperparameters["reg_parameter"]
        self.n_epochs = hyperparameters["n_epochs"]
        self.n_factors = hyperparameters["n_factors"]
        self.early_stopping = hyperparameters.get('early_stopping', True)
        self.rec_mtx = None

        self.alg = SVDpp(n_factors = self.n_factors,
                         n_epochs = self.n_epochs,
                         lr_all = self.lr,
                         reg_all = self.reg)
        self.alg.fit(full_surp_trainset)

    def train(self) -> float:
        return 0

    # predicts on the test set and return the score
    def test(self) -> float:
        preds= [self.alg.predict(row["userID"], row["itemID"]).est for _, row in self.test_data.iterrows()]
        self.test_score = self.get_score(preds, self.test_data["rating"])
        return self.test_score

    # predicts the reviews for the actual unknowns
    def predict(self, to_predict):
        predict_data = self.format_data(to_predict)
        preds= [self.alg.predict(row["userID"], row["itemID"]).est for _, row in predict_data.iterrows()]
        to_predict["Prediction"] = preds
        return to_predict

    def format_data(self, data):
        # reformats into dataframe with columns u_id, i_id, and rating
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values, 2, axis=-1)]

        predictions = data.Prediction.values
        return pd.DataFrame({"userID": users, "itemID": movies, "rating": predictions})

    def rmse(self, x, y): return math.sqrt(mean_squared_error(x, y))

    def get_score(self, predictions, target_values):
        return self.rmse(predictions, target_values)
