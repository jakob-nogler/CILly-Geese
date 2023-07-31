import numpy as np
import random
import torch as T


class Model():
    """A template class for our collaborative filtering models
    """

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        self.train_set = train_set
        self.test_set = test_set
        self.hyperparameters = hyperparameters
        #set random seeds
        np.random.seed(hyperparameters['seed'])
        T.manual_seed(hyperparameters['seed'])
        random.seed(hyperparameters['seed'])


    def train(self) -> float:
        """Performs one epoch of the training"""
        pass

    def test(self) -> float:
        """Computes the validation score"""
        pass

    def predict(self, to_predict):
        """Predicts the reviews for the to_predict (in the kaggle format)"""
        pass

    def save(self, path: str):
        """Saves the model under the specified path"""
        pass

    def load(self, path: str):
        """Loads the model from the specified path"""
        pass
