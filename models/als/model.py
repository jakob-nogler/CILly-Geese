from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
from ..model import Model
from scipy.sparse.linalg import svds


class ALSModel(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        self.train_users, self.train_movies, self.train_predictions = self.extract_data(
            train_set)
        self.test_users, self.test_movies, self.test_predictions = self.extract_data(
            test_set)

        self.rank = hyperparameters["rank"]
        self.num_iters = hyperparameters["num_iterations"]
        self.lam = hyperparameters["reg_parameter"]
        self.num_svd_runs = hyperparameters.get("num_svd_runs", 3)
        self.lr = hyperparameters.get("svd_lr", 0.1)
        self.use_iSVD = hyperparameters.get("iSVD", False) # determines whether we use iSVD or proj_SVD
        self.transpose = hyperparameters.get("transpose", False)
        self.rec_mtx = None

    def train(self) -> float:
        if self.rec_mtx is None:
            self.rec_mtx = self.ALS(
                self.train_users, self.train_movies, self.train_predictions)
            preds = self.extract_prediction_from_full_matrix(
                self.rec_mtx, self.test_users, self.test_movies)

            # whole training
            self.test_score = self.get_score(preds, self.test_predictions)
        return 0

    # predicts on the test set and return the score
    def test(self) -> float:
        return self.test_score

    # predicts the reviews for the actual unknowns
    def predict(self, to_predict):
        users, movies, _ = self.extract_data(to_predict)
        predictions = self.extract_prediction_from_full_matrix(
            self.rec_mtx, users, movies)
        to_predict['Prediction'] = predictions
        return to_predict

    def extract_data(self, data):
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        predictions = data.Prediction.values
        return users, movies, predictions

    def extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        assert (len(users) == len(movies)
                ), "users-movies combinations specified should have equal length"
        predictions = np.zeros(len(users))

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]

        return predictions

    def rmse(self, x, y): return math.sqrt(mean_squared_error(x, y))

    def get_score(self, predictions, target_values):
        return self.rmse(predictions, target_values)
    
    def proj_SVD(self, A, mask, lr=0.1, rank=10, num_iters=10):
        U = np.random.uniform(low=-1.0, high=1.0, size=(A.shape[0], rank))
        V = np.random.uniform(low=-1.0, high=1.0, size=(rank, A.shape[1]))
        A_curr = np.zeros((A.shape[0], A.shape[1]))
        for _ in range(num_iters):
            diff = np.multiply(np.subtract(A, A_curr), mask)
            pre_svd = A_curr + lr*diff
            U, S, Vt = svds(pre_svd, k=rank)
            S = np.diag(S)
            U = U.dot(np.sqrt(S))
            V = np.sqrt(S).dot(Vt)
            A_curr = U.dot(V)
        return U, V, A_curr     

    def iSVD(self, A, mask, rank=10, num_iters=3):
        U = np.random.uniform(low=-1.0, high=1.0, size=(A.shape[0], rank))
        V = np.random.uniform(low=-1.0, high=1.0, size=(rank, A.shape[1]))
        A_curr = A
        for _ in range(num_iters):
            U_init, S, Vt = svds(A_curr, k=rank)
            S = np.diag(S)
            U = U_init.dot(np.sqrt(S))
            V = np.sqrt(S).dot(Vt)
            A_curr = np.multiply(A, mask) + np.multiply(U@V, 1-mask)
        return U, V, A_curr

    def ALS(self, users, movies, preds):
        rows, cols = (10000, 1000)
        data = np.zeros((rows, cols))
        # 0 -> unobserved value, 1->observed value
        mask = np.zeros((rows, cols))

        for user, movie, pred in zip(users, movies, preds):
            data[user][movie] = pred
            mask[user][movie] = 1

        if self.transpose:
            axis = 1
        else:
            axis = 0

        means = np.nanmean(np.where(data != 0, data, np.nan), axis=axis)
        stds = np.nanstd(np.where(data != 0, data, np.nan), axis=axis)

        if self.transpose:
            A = ((data.transpose() - means)/stds).transpose()
        else:
            A = (data-means)/stds
        A = np.multiply(A, mask)

        # SVD to initialize
        U, V = None, None
        if self.use_iSVD:
            U, V, _ = self.iSVD(A, mask, self.rank, self.num_svd_runs)
        else:
            U, V, _ = self.proj_SVD(A, mask, self.lr, self.rank, self.num_svd_runs)

        for _ in range(self.num_iters):
            # update to Y
            B = U.transpose() @ A
            Q = np.ndarray((rows, self.rank, self.rank))
            for row in range(rows):
                Q[row] = np.outer(U[row, :].transpose(), U[row, :])
            for col in range(cols):
                sum_r1_mat = 0
                for row in range(rows):
                    if mask[row, col]:
                        sum_r1_mat += Q[row, :, :]
                inv = np.linalg.inv(
                    sum_r1_mat + self.lam * np.identity(self.rank))
                V[:, col] = inv @ B[:, col]

            # update to X
            B = V @ A.transpose()
            Q = np.ndarray((cols, self.rank, self.rank))
            for col in range(cols):
                Q[col] = np.outer(V[:, col], V[:, col].transpose())
            for row in range(rows):
                sum_r1_mat = 0
                for col in range(cols):
                    if mask[row, col]:
                        sum_r1_mat += Q[col, :, :]
                inv = np.linalg.inv(
                    sum_r1_mat + self.lam * np.identity(self.rank))
                U[row, :] = inv @ B[:, row]

        # store decomposistion for lower memory saving
        if self.transpose:
            output = ((U@V).transpose() * stds + means).transpose()
        else:
            output = (U@V)*stds + means
        rec_mtx = np.clip(output, 1, 5)
        return rec_mtx

    def save(self, path: str):
        np.save(path, self.rec_mtx)

    def load(self, path: str):
        self.rec_mtx = np.load(path)
