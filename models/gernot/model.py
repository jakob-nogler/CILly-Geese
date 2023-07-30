from ..model import Model
import torch
import math
from torch import nn
from torch import optim
from .network import Network
from .dataloader import *
from sklearn.metrics import mean_squared_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number_of_users, number_of_movies = (10000, 1000)


class Gernot(Model):

    def __init__(self, hyperparameters: dict, train_set, test_set) -> None:
        super().__init__(hyperparameters, train_set, test_set)

        self.lr = self.hyperparameters.get('learning_rate', 0.01)
        self.weight_decay = self.hyperparameters.get('weight_decay', 0.001)
        self.batch_size = self.hyperparameters.get('batch_size', 128)
        self.embedding_dim = self.hyperparameters.get('embedding_dim', 16)
        self.hidden_dimension = self.hyperparameters.get(
            "hidden_dimension", 64)
        self.activation = self.hyperparameters.get("activation", "relu")
        self.n_layers = self.hyperparameters.get('n_layers', 10)
        self.drop_probability = self.hyperparameters.get('drop_probability', 0)
        self.patience = self.hyperparameters.get('patience', 5)
        self.normalize = self.hyperparameters.get("normalize", "items")

        self.loss_fn = nn.MSELoss().to(device)

        self.preprocess_data()

        self.training_mask, train_sp = make_training_mask(
            self.train_dataloader, number_of_users, number_of_movies)
        self.sparse_graph = get_sparse_graph(
            self.training_mask, number_of_users, number_of_movies)

        if self.normalize == 'users':
            norm_means, norm_stds = self.compute_csr_stats(train_sp)
        elif self.normalize == 'items':
            norm_means, norm_stds = self.compute_csr_stats(train_sp.T)
        else:
            norm_means, norm_stds = np.zeros(
                (number_of_users)), np.ones((number_of_users))

        self.model = Network(sparse_graph=self.sparse_graph,
                             num_users=number_of_users,
                             num_items=number_of_movies,
                             activation=self.activation,
                             user_dataloader=DataLoader(
                                 self.matrix, batch_size=self.batch_size),
                             movie_dataloader=DataLoader(
                                 self.matrixT, batch_size=self.batch_size),
                             hidden_dimension=self.hidden_dimension,
                             embedding_dim=self.embedding_dim,
                             n_layers=self.n_layers,
                             use_dropout=self.drop_probability > 1e-9,
                             keep_probability=1 - self.drop_probability,
                             norm_mode=self.normalize,
                             norm_means=norm_means,
                             norm_stds=norm_stds).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                 factor=0.5,
                                                                 patience=self.patience)

    # whole training
    def train(self) -> float:
        self.model.train()
        total_loss = 0.
        for _, (users, movies, ratings) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            y_pred = self.model(users, movies)
            loss = self.loss_fn(y_pred, ratings)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.model.eval()
        val_score = self.test()
        self.lr_scheduler.step(val_score)
        return total_loss

    # predicts on the test set and return the score
    def test(self) -> float:
        '''Uses RMSE to compute validation loss'''
        total_items = 0
        all_predictions = []
        for users, movies in self.test_dataloader:
            total_items += len(users)
            with torch.no_grad():
                y_pred = self.model(users, movies)
                all_predictions.append(y_pred)

        all_predictions = torch.cat(all_predictions)
        return self.get_score(all_predictions.cpu().numpy(), self.test_predictions.cpu().numpy())

    def rmse(self, x, y): return math.sqrt(mean_squared_error(x, y))

    def get_score(self, predictions, target_values):
        return self.rmse(predictions, target_values)
    # predicts the reviews for the

    def predict(self, to_predict):
        dataloader = get_dataloader(to_predict, batch_size=self.batch_size)
        with torch.no_grad():
            all_predictions = []
            for users, movies, _ in dataloader:
                pred = self.model(users, movies)
                all_predictions.append(pred)
        all_predictions = torch.cat(all_predictions)
        to_predict["Prediction"] = all_predictions.cpu().numpy()
        return to_predict

    def preprocess_data(self):

        test_users, test_movies, test_predictions = self.extract_data(
            self.test_set)

        test_users = torch.tensor(test_users, device=device)
        test_movies = torch.tensor(test_movies, device=device)
        self.test_predictions = torch.tensor(test_predictions, device=device)

        self.test_dataloader = DataLoader(TensorDataset(
            test_users, test_movies), batch_size=self.batch_size)

        users, movies, predictions = self.extract_data(self.train_set)

        matrix = np.zeros(
            (number_of_users, number_of_movies), dtype=np.float32)
        mask = np.zeros((number_of_users, number_of_movies), dtype=np.int32)

        for user, movie, prediction in zip(users, movies, predictions):
            matrix[user][movie] = prediction
            mask[user][movie] = 1

        self.matrix = torch.tensor(matrix, device=device)
        self.matrixT = torch.tensor(matrix.transpose(), device=device)
        self.mask = torch.tensor(mask, device=device)

        train_users = torch.tensor(users, device=device)
        train_movies = torch.tensor(movies, device=device)
        train_predictions = torch.tensor(
            predictions, device=device, dtype=torch.float32)
        self.train_dataloader = DataLoader(TensorDataset(
            train_users, train_movies, train_predictions), batch_size=self.batch_size)

    def extract_data(self, data):
        """Reads the data from the pd.DataFrame in the kaggle format
        """
        users, movies = [np.squeeze(arr) for arr in np.split(
            data.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

        predictions = data.Prediction.values
        return users, movies, predictions

    # saves the model under the specified path
    def save(self, path: str):
        pass

    # loads the model from the specified path
    def load(self, path: str):
        pass

    def compute_csr_stats(self, mat: "sparse.csr_matrix"):
        means = np.zeros((mat.shape[0],))
        standard_devs = np.zeros((mat.shape[0],))

        for i in range(mat.shape[0]):
            my_row = mat.getrow(i).toarray()[0]
            my_row = my_row[my_row > 0]
            means[i] = my_row.mean()
            standard_devs[i] = my_row.std()

        return means, standard_devs
