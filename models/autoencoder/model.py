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


class Autoencoder(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        # read some hyperparameters
        self.predict_interaction = self.hyperparameters.get(
            "predict_interactions", False)
        self.indicate_interactions = self.hyperparameters.get(
            "indicate_interactions", True)
        self.enhancement_factor = self.hyperparameters.get(
            "enhancement_factor", 0)
        self.num_refeeds = self.hyperparameters.get("refeeds", 0)
        self.one_hot_reviews = self.hyperparameters.get(
            "one_hot_reviews", False)
        self.transpose = self.hyperparameters.get("transpose", False)
        self.optimizer_name = self.hyperparameters.get("optimizer", "adam")

        self.batch_size = self.hyperparameters.get("batch_size", 256)

        # reformats the data
        self.preprocess_data()

        # initialize the model and the optimizer etc.
        self.model = NeuralNet(
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            hidden_dimension=hyperparameters.get("hidden_dimension", 128),
            encoded_dimension=hyperparameters.get("encoded_dimension", 24),
            num_hidden_encoder=hyperparameters.get("num_hidden_encoder", 1),
            num_hidden_decoder=hyperparameters.get("num_hidden_decoder"),
            drop_probability=hyperparameters.get("drop_probability", 0),
            activation=hyperparameters.get("activation", "relu")
        ).to(device)

        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=hyperparameters.get(
                                            'learning_rate', 1e-3),
                                        weight_decay=hyperparameters.get('weight_decay', 1e-3))
        elif self.optimizer_name == "momentum":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=hyperparameters.get(
                                           'learning_rate', 1e-3),
                                       weight_decay=hyperparameters.get(
                                           'weight_decay', 1e-3),
                                       momentum=hyperparameters.get("momentum", 0.9))

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=20,
            factor=0.5,
        )

        self.dataloader = DataLoader(
            TensorDataset(self.input_matrix,
                          self.output_matrix, self.output_mask),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.epoch = -1
        self.last_tested_epoch = -1
        self.last_test_result = -1

    def train(self):
        self.epoch = self.epoch + 1

        loss_sum = 0
        loss_count = 0
        self.model.train()
        self.model.set_training()
        for data_batch, review_batch, mask_batch in self.dataloader:
            # predict and do backprop
            num_rows = data_batch.shape[0]

            self.optimizer.zero_grad()

            reconstructed = self.model(data_batch)

            loss = self.loss_function(review_batch, reconstructed, mask_batch)
            loss_sum += loss.item()
            loss_count += 1
            loss.backward()
            self.optimizer.step()

            if not self.one_hot_reviews:
                for _ in range(self.num_refeeds):
                    self.optimizer.zero_grad()
                    # treat the previous output as the new input and train the network on it

                    # drop some of the reviews in the output based on the refeed_density
                    mask_prob = T.ones((num_rows, self.output_dimension), device=device) * \
                        self.hyperparameters.get('refeed_density', 0.5)
                    mask = T.bernoulli(mask_prob)

                    self.optimizer.zero_grad()
                    x = T.repeat_interleave(reconstructed.detach() * mask, 2)
                    x = x.reshape((-1, self.input_dimension))

                    if self.predict_interaction:
                        inx = T.zeros((num_rows, self.output_dimension),
                                      dtype=T.bool, device=device)
                        for i in range(self.output_dimension // 2):
                            inx[:, 2*i + 1] = 1
                        x[inx] = mask.reshape(-1)

                    reconstructed_1 = self.model(x)

                    loss = self.loss_function(
                        reconstructed.detach(), reconstructed_1, mask)
                    loss_sum += loss.item()
                    loss_count += 1
                    loss.backward()
                    self.optimizer.step()
                    reconstructed = reconstructed_1.detach()

        self.model.eval()
        self.scheduler.step(self.test())

        return loss_sum

    # predicts on the test set and return the score
    def test(self) -> float:
        if self.epoch == self.last_tested_epoch:
            return self.last_test_result

        # recostruct the whole review matrix and then read the specific predictions from it
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
        # recostruct the whole review matrix and then read the specific predictions from it
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

        if self.transpose:
            self.number_rows = number_of_movies
            self.number_columns = number_of_users
            data = np.transpose(data)
            mask = np.transpose(mask)
        else:
            self.number_rows = number_of_users
            self.number_columns = number_of_movies

        # save separate matrices for input and output
        self.output_matrix = np.zeros(
            (self.number_rows, self.number_columns), dtype=np.float32)
        self.output_mask = np.ones(
            (self.number_rows, self.number_columns), dtype=np.int32)

        # matrices for the enhanced data set -> created by dropping some existing reviews with a specified probability
        enhanced_output_matrix = np.zeros(
            (self.enhancement_factor * self.number_rows, self.number_columns), dtype=np.float32
        )
        enhanced_output_mask = np.ones(
            (self.enhancement_factor * self.number_rows, self.number_columns), dtype=np.int32
        )

        self.output_dimension = self.number_columns
        if self.predict_interaction:
            self.output_dimension *= 2

        # create the input matrix as specified in the hyperparameters
        if self.one_hot_reviews:
            # indicate the reviews by boolean input vector
            self.input_matrix = np.zeros(
                (self.number_rows, self.number_columns * 6), dtype=np.float32)
            enhanced_input_matrix = np.zeros(
                (self.number_rows*self.enhancement_factor, self.number_columns * 6), dtype=np.float32)
            self.input_dimension = self.number_columns * 6
            for row in range(self.number_rows):
                non_zeros = np.nonzero(mask[row])[0]
                random_draws = len(non_zeros) // 2
                for col in range(self.number_columns):
                    if not mask[row][col]:
                        if self.indicate_interactions:
                            self.input_matrix[row, 6 * col + 5] = 1
                        self.output_mask[row, col] = 0
                    else:
                        self.input_matrix[row, 6*col +
                                          int(data[row][col]) - 1] = 1
                        self.output_matrix[row, col] = data[row][col]

                # enhance the data
                for enh in range(self.enhancement_factor):
                    dropped_indices = []
                    for _ in range(random_draws):
                        dropped_indices.append(np.random.choice(non_zeros))

                    for col in range(self.number_columns):
                        if not mask[row][col]:
                            if self.indicate_interactions:
                                enhanced_input_matrix[self.enhancement_factor *
                                                      row + enh, 6 * col + 5] = 1
                            enhanced_output_mask[self.enhancement_factor *
                                                 row + enh, col] = 0
                        elif col in dropped_indices:
                            if self.indicate_interactions:
                                enhanced_input_matrix[self.enhancement_factor *
                                                      row + enh, 6 * col + 5] = 1
                        else:
                            enhanced_input_matrix[self.enhancement_factor*row + enh, 6*col +
                                                  int(data[row][col]) - 1] = 1
                            enhanced_output_matrix[self.enhancement_factor *
                                                   row + enh, col] = data[row][col]
        else:
            # indicate the review by a single number in the input
            self.input_matrix = np.zeros(
                (self.number_rows, self.number_columns * 2), dtype=np.float32)
            enhanced_input_matrix = np.zeros(
                (self.number_rows*self.enhancement_factor, self.number_columns * 2), dtype=np.float32)
            self.input_dimension = self.number_columns * 2
            for row in range(self.number_rows):
                non_zeros = np.nonzero(mask[row])[0]
                random_draws = len(non_zeros) // 2
                for col in range(self.number_columns):
                    if not mask[row][col]:
                        if self.indicate_interactions:
                            self.input_matrix[row, 2 * col + 1] = 1
                        self.output_mask[row, col] = 0
                    else:
                        self.input_matrix[row, 2*col] = data[row][col]
                        self.output_matrix[row, col] = data[row][col]

                # enhance the data
                for enh in range(self.enhancement_factor):
                    dropped_indices = []
                    for _ in range(random_draws):
                        dropped_indices.append(np.random.choice(non_zeros))

                    for col in range(self.number_columns):
                        if not mask[row][col]:
                            if self.indicate_interactions:
                                enhanced_input_matrix[row, 2 * col + 1] = 1
                            enhanced_output_mask[row, col] = 0
                        elif col in dropped_indices:
                            if self.indicate_interactions:
                                enhanced_input_matrix[row, 2 * col + 1] = 1
                        else:
                            enhanced_input_matrix[row, 2*col] = data[row][col]
                            enhanced_output_matrix[row, col] = data[row][col]

        # add the enhanced data to the data set
        self.input_matrix = np.concatenate(
            (self.input_matrix, enhanced_input_matrix), axis=0)
        self.output_matrix = np.concatenate(
            (self.output_matrix, enhanced_output_matrix), axis=0)
        self.output_mask = np.concatenate(
            (self.output_mask, enhanced_output_mask), axis=0)

        self.number_rows = (self.enhancement_factor + 1) * self.number_rows

        # save the matrices for training
        self.input_matrix = T.tensor(self.input_matrix, device=device).float()
        self.output_matrix = T.tensor(
            self.output_matrix, device=device).float()
        self.output_mask = T.tensor(self.output_mask, device=device)

    def extract_data(self, data):
        """Reads the data from the pd.DataFrame in the kaggle format
        """
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        predictions = data.Prediction.values
        return users, movies, predictions

    def loss_function(self, original, reconstructed, mask):
        if self.predict_interaction:
            # the network should learn to predict whether the user will interact with the movie or not too
            reconstructed, interactions = reconstructed.split(
                self.number_columns, dim=1)
            return T.mean(mask * (original - reconstructed) ** 2 + (mask - interactions)**2)
        else:
            # simple mse
            return T.mean(mask * (original - reconstructed) ** 2)

    def reconstruct_whole_matrix(self,):
        """Predicts a review for each data-movie pair"""
        data_reconstructed = np.zeros((self.number_rows, self.number_columns))
        self.model.set_inference()

        with T.no_grad():
            for i in range(0, self.number_rows, self.batch_size):
                upper_bound = min(
                    i + self.batch_size, self.number_rows)
                predictions = self.model(self.input_matrix[i:upper_bound])
                if self.predict_interaction:
                    predictions, _ = predictions.split(
                        self.number_columns, dim=1)

                reconstructed = predictions
                data_reconstructed[i:upper_bound] = reconstructed.detach(
                ).cpu().numpy()

        if self.transpose:
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
