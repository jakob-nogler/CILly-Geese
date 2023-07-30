import pandas as pd
import numpy as np
import pandas as pd
from scipy import sparse as sps
from typing import Dict, List, Union


import myfm
from myfm import RelationBlock
from myfm.utils.callbacks import (
    RegressionCallback,
)
from myfm.utils.encoders import CategoryValueToSparseEncoder

from ..model import Model


class BayesianSVD(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        self.iterations = self.hyperparameters.get("num_epochs", 100)
        self.rank = self.hyperparameters.get("rank", 30)

        self.user_interactions = self.hyperparameters.get(
            "user_interactions", True)
        self.movie_interactions = self.hyperparameters.get(
            "movie_interactions", True)

        self.train_set = self.convert_dataframe(self.train_set)
        self.test_set = self.convert_dataframe(self.test_set)

        self.prepare_implicit_data()
        self.train_blocks = self.prepare_blocks(self.train_set)
        self.test_blocks = self.prepare_blocks(self.test_set)

        feature_group_sizes = []

        feature_group_sizes.append(len(self.user_to_internal))  # user ids
        if self.user_interactions:
            # all movies which a user watched
            feature_group_sizes.append(len(self.movie_to_internal))
        feature_group_sizes.append(len(self.movie_to_internal))  # movie ids
        if self.movie_interactions:
            feature_group_sizes.append(
                # all the users who watched a movies
                len(self.user_to_internal)
            )
        grouping = [i for i, size in enumerate(
            feature_group_sizes) for _ in range(size)]

        self.fm = myfm.MyFMRegressor(
            rank=self.rank, random_seed=self.hyperparameters.get("seed", 42))
        callback = None
        if True:
            callback = RegressionCallback(
                n_iter=self.iterations,
                X_test=None,
                y_test=self.test_set.rating.values,
                X_rel_test=self.test_blocks,
                clip_min=self.train_set.rating.min(),
                clip_max=self.train_set.rating.max(),
            )

        self.fm.fit(
            None,
            self.train_set.rating.values,
            X_rel=self.train_blocks,
            grouping=grouping,
            n_iter=self.iterations,
            n_kept_samples=self.iterations,
            callback=callback,
        )

        prediction = self.fm.predict(None, self.test_blocks)

        self.test_score = (
            (self.test_set.rating - prediction) ** 2).mean() ** 0.5

    def train(self) -> float:
        return 0

    def test(self) -> float:
        return self.test_score

    def predict(self, to_predict):
        converted = self.convert_dataframe(to_predict)
        blocks = self.prepare_blocks(converted)
        predictions = self.fm.predict(None, blocks)

        to_predict["Prediction"] = predictions
        return to_predict

    def convert_dataframe(self, data):
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int), 2, axis=-1)]

        predictions = data.Prediction.values
        # roundabout way of reformatting but pandas scares me
        return pd.DataFrame({"user_id": users, "movie_id": movies, "rating": predictions})

    def prepare_implicit_data(self):
        implicit_data_source = pd.read_csv("./data/user_movie_pairs.csv")
        self.user_to_internal = CategoryValueToSparseEncoder[int](
            implicit_data_source.user_id.values
        )
        self.movie_to_internal = CategoryValueToSparseEncoder[int](
            implicit_data_source.movie_id.values
        )
        self.movie_vs_watched: Dict[int, List[int]] = dict()
        self.user_vs_watched: Dict[int, List[int]] = dict()

        for row in implicit_data_source.itertuples():
            user_id: int = row.user_id
            movie_id: int = row.movie_id
            self.movie_vs_watched.setdefault(movie_id, list()).append(user_id)
            self.user_vs_watched.setdefault(user_id, list()).append(movie_id)

    def prepare_blocks(self, data_df: "pd.DataFrame"):
        blocks: List[RelationBlock] = []

        unique_users, user_map = np.unique(
            data_df.user_id, return_inverse=True)
        blocks.append(RelationBlock(
            user_map, self.augment_user_id(unique_users)))
        unique_movies, movie_map = np.unique(
            data_df.movie_id, return_inverse=True)
        blocks.append(RelationBlock(
            movie_map, self.augment_movie_id(unique_movies)))

        return blocks

    def augment_user_id(self, user_ids: List[int]) -> sps.csr_matrix:
        X = self.user_to_internal.to_sparse(user_ids)
        if not self.user_interactions:
            return X
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []
        for index, user_id in enumerate(user_ids):
            watched_movies = self.user_vs_watched.get(user_id, [])
            normalizer = 1 / max(len(watched_movies), 1) ** 0.5
            for mid in watched_movies:
                data.append(normalizer)
                col.append(self.movie_to_internal[mid])
                row.append(index)
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(user_ids), len(self.movie_to_internal)),
                ),
            ],
            format="csr",
        )

    def augment_movie_id(self, movie_ids: List[int]) -> sps.csr_matrix:
        X = self.movie_to_internal.to_sparse(movie_ids)
        if not self.movie_interactions:
            return X
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []
        for index, movie_id in enumerate(movie_ids):
            watched_users = self.movie_vs_watched.get(movie_id, [])
            normalizer = 1 / max(len(watched_users), 1) ** 0.5
            for uid in watched_users:
                data.append(normalizer)
                row.append(index)
                col.append(self.user_to_internal[uid])
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(movie_ids), len(self.user_to_internal)),
                ),
            ]
        )
