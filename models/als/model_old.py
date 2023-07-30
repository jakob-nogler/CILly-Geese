from sklearn.metrics import mean_squared_error
import math
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from ..model import Model

import pandas as pd
import findspark
findspark.init()


class ALSModel(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        self.spark = SparkSession.Builder().master(
            'local').appName('ALS').config('spark.ui.port', '4050').getOrCreate()
        self.spark.sparkContext.setLogLevel('OFF')

        train_users, train_movies, train_predictions = self.extract_data(
            train_set)
        test_users, test_movies, test_predictions = self.extract_data(test_set)

        train_data = self.spark.createDataFrame(pd.DataFrame({
            'User': train_users,
            'Movie': train_movies,
            'Prediction': train_predictions
        }))
        test_data = self.spark.createDataFrame(pd.DataFrame({
            'User': test_users,
            'Movie': test_movies,
        }))

        als = ALS(rank=hyperparameters['rank'],
                  maxIter=hyperparameters['num_iterations'],
                  regParam=hyperparameters['reg_parameter'],
                  seed=hyperparameters['seed'],
                  userCol="User",
                  itemCol="Movie",
                  ratingCol="Prediction",
                  coldStartStrategy="drop",
                  )

        self.model = als.fit(train_data)

        # whole training
        test_predictions_model = self.model.transform(test_data)
        self.test_score = self.get_score(test_predictions_model.toPandas()[
                                         'prediction'].to_numpy(), test_predictions)

    def train(self) -> float:
        return 0

    # predicts on the test set and return the score
    def test(self) -> float:
        return self.test_score

    # predicts the reviews for the
    def predict(self, to_predict):
        users, movies, _ = self.extract_data(to_predict)
        data = self.spark.createDataFrame(pd.DataFrame({
            'User': users,
            'Movie': movies,
        }))
        predictions = self.model.transform(data)
        to_predict['Prediction'] = predictions.toPandas()['prediction']
        return to_predict

    def extract_data(self, data):
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        predictions = data.Prediction.values
        return users, movies, predictions

    def rmse(self, x, y): return math.sqrt(mean_squared_error(x, y))

    def get_score(self, predictions, target_values):
        return self.rmse(predictions, target_values)
