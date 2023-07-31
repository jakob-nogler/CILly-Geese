import copy
import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error
import torch.optim as optim
import torch
import json
import time
import wandb

from train_model import get_trained_model
from utils.combinator import Combinator
from sklearn.model_selection import train_test_split
from utils.early_stopping import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")


def start_training(config: "dict", verbose: bool = False, save_log: bool = False, use_full_data: bool = False, save_models=False, output_config_name: str = None, output_prediction_name: str = None, use_wandb: bool = False, sweep_id: str = None, save_validation: bool = False, almost_full_data: bool = False, movielens: bool = False):
    """Trains the models as specified by the config dict
    At the end combines the results of all the models that passed their corresponding baselines in an ensemble and saves the results to a file under results/

    Args:
        config (dict): dictionary with specifying what models are to be trained
        verbose (bool, optional): print detailed training information to console
        save_log (bool, optional): save logs with training details to file
        use_full_data (bool, optional): use full data for training (no train - validation split)
        save_models (bool, optional): saves the trained models to files
        output_config_name (str, optional): file name for the config file with hyperparameters of models trained in this process
        output_prediction_name (str, optional): file name for the final predictions
        use_wandb (bool, optional): Use the wandb sweeps framework for hyperparameter search - requires wandb login
        sweep_id (str, optional): Sweed id of the sweep to be resumed (o/w creates a new sweep)
        save_validation (bool, optional): Saves the predictions on the validation set.
    """
    full_config = copy.deepcopy(config)
    config_list = full_config["config_list"]

    total_iterations = 0
    for config in config_list:
        total_iterations += config["num_repetitions"]

    # load training data
    if movielens:
        train_data = pd.read_csv("./data/movielens_train.csv")
        test_data = pd.read_csv("./data/movielens_test.csv")
    elif use_full_data:
        train_data = pd.read_csv("./data/data_full.csv")
        test_data = train_data
    elif almost_full_data:
        train_data = pd.read_csv("./data/data_full_train.csv")
        test_data = pd.read_csv("./data/data_full_test.csv")
    else:
        train_data = pd.read_csv("./data/data_train.csv")
        test_data = pd.read_csv("./data/data_test.csv")

    predictions_template = pd.read_csv("./data/prediction_template.csv")

    running = True

    output_configs = []

    test_predictions = []
    predictions = []

    # helper function to train a single model, needed to pass to wandb agent
    def model_iteration(config: dict):
        """Trains the model as specified in the config dictionary

        Args:
            config (dict, optional): information on the model to be trained
        """
        model, hyperparameters, log = get_trained_model(
            config, train_data, test_data, verbose=verbose, save_model=save_models, use_wandb=use_wandb, full_data=use_full_data)

        # compute validation score
        test_prediction = copy.deepcopy(test_data)
        test_prediction = model.predict(test_prediction)

        test_score = compute_score(
            test_prediction["Prediction"].to_numpy(), test_data["Prediction"].to_numpy())

        print(
            f"{config['model']} | test score {test_score} | left: {total_iterations}")
        if verbose:
            print_hyperparameters(hyperparameters)

        # if model passed the baseline - add it to the ensemble
        if test_score <= config["baseline"]:
            test_predictions.append(test_prediction)

            prediction = copy.deepcopy(predictions_template)
            prediction = model.predict(prediction)
            predictions.append(prediction)

        # save the config to with the hyperparameters drawn to the output config file
        new_config = copy.deepcopy(config)
        new_config["hyperparameters"] = hyperparameters
        new_config["test_score"] = test_score
        new_config["num_repetitions"] = 1
        if save_log:
            new_config["log"] = log
        output_configs.append(new_config)
        # tidy up
        del model
        torch.cuda.empty_cache()

    # train models until no more left in the config file
    while running:
        running = False

        for config in config_list:

            # check if enough runs of this config have already been made
            if config["num_repetitions"] <= 0:
                continue

            # check if this model was already trained and doesn't pass the baseline
            if "test_score" in config and config["test_score"] > config["baseline"]:
                output_configs.append(config)
                continue

            if use_wandb:
                # create a new sweep if no sweep_id is specified
                if not sweep_id and not "sweep_id" in config:
                    sweep_id = wandb.sweep(
                        sweep=config["sweep_configuration"],
                        project=config["model"]
                    )
                elif not sweep_id and "sweep_id" in config:
                    sweep_id = config["sweep_id"]

                # helper function to pass the config file
                def fn():
                    model_iteration(config)

                # start the wandb agent (with num_repetition many runs)
                wandb.agent(
                    sweep_id, function=fn, count=config["num_repetitions"], project=config["model"])
                config["num_repetitions"] = 0
            else:
                # train the model manually -> only single iteration
                model_iteration(config)
                total_iterations -= 1

                running = True
                config["num_repetitions"] -= 1

                # save the checkpoint for later restart
                save_checkpoint(full_config, output_configs)

    # check if any models are part of the ensemble
    if len(test_predictions) == 0:
        print("No models passed the baseline. Finishing...")
        return

    # reformat data for combination weight training
    test_predictions_combined = np.zeros(
        (len(test_predictions), test_predictions[0]["Prediction"].shape[0]))
    for i in range(len(test_predictions)):
        test_predictions_combined[i,
                                  :] = test_predictions[i]["Prediction"].to_numpy()

    # reformat the final predictions for combination
    predictions_combined = np.zeros(
        (len(predictions), predictions[0]["Prediction"].shape[0]))
    for i in range(len(predictions)):
        predictions_combined[i, :] = predictions[i]["Prediction"].to_numpy()

    # check if combination weights have already been specified -> otherwise train them
    if "combination_weights" in full_config and full_config["combination_weights"]:
        weights = np.array(full_config["combination_weights"])
    else:
        weights = train_combination_weights(
            test_predictions_combined, test_data["Prediction"].to_numpy(), verbose=verbose, full_data=use_full_data)

    # combine the predictions on the validation set of the models in the ensemble
    combined_test = combine_results(test_predictions_combined, weights)
    # and compute the score of the combined prediction
    test_score = compute_score(
        combined_test, test_data["Prediction"].to_numpy())
    print(f"Overall test score {test_score}")

    # combine the prediction for kaggle
    combined_prediction = combine_results(predictions_combined, weights)
    predictions_template["Prediction"] = combined_prediction

    # save the config of this training process to file
    output_config = {}
    output_config["combination_weights"] = weights.tolist()
    output_config["config_list"] = output_configs
    output_config["test_score"] = test_score

    save_config(output_config, output_config_name)

    if save_validation:
        test_data["Prediction"] = combined_test
        save_test_predictions(test_data, test_score, output_prediction_name)

    # save the predictions of the ensemble to file -> can be submitted to kaggle
    save_predictions(predictions_template, test_score, output_prediction_name)

    return combined_prediction


def combine_results(predictions: "np.array", weights: "np.array") -> "np.array":
    """Combines the predictions of the models in the ensemble according to the specified weights
    Args:
        predictions (np.array): predictions of the single models in the ensemble
        weights (np.array): pre-trained combination weights

    Returns:
        np.array: combined prediction of the ensemble
    """
    predictions = torch.tensor(predictions, device=device)
    weights = torch.tensor(weights, device=device)

    model = Combinator(predictions.shape[0]).to(device)
    model.set_weights(weights)

    return model(predictions).detach().cpu().numpy()


def train_combination_weights(predictions: "np.array", target_values: "np.array", verbose: bool = False, full_data: bool = False) -> "np.array":
    """Trains the weights for an affine combination of the predictions of the model in the ensemble
    Args:
        predictions (np.array): predictions on the validation set
        target_values (np.array): true validation set values

    Returns:
        np.array: combination weights
    """

    # pre-process data
    predictions = predictions.transpose()
    train_predictions, test_predictions, train_target, test_target = train_test_split(
        predictions, target_values, test_size=0.5)
    predictions = predictions.transpose()
    train_predictions = train_predictions.transpose()
    test_predictions = test_predictions.transpose()

    train_predictions = torch.tensor(train_predictions, device=device).float()
    test_predictions = torch.tensor(test_predictions, device=device).float()
    train_target = torch.tensor(train_target, device=device).float()
    test_target = torch.tensor(test_target, device=device).float()

    # the model performs an affine combination of the input
    model = Combinator(predictions.shape[0]).to(device)

    stopper = EarlyStopping(patience=15, delta=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=10,
        factor=0.5,
    )

    loss = torch.nn.MSELoss()

    # train the weights for the affine combination
    while not stopper.early_stop and not full_data:
        optimizer.zero_grad()
        pred = model(train_predictions)
        l = loss(pred, train_target)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            pred = model(test_predictions)
            validation = loss(pred, test_target)
            scheduler.step(validation.item())
            stopper.step(validation.item())
            if verbose:
                print(f"Current validation loss {validation.item()}")

    return model.get_weights().detach().cpu().numpy()


def compute_score(predictions: "np.array", target_values: "np.array") -> float:
    """Computes rmse
    """
    return math.sqrt(mean_squared_error(target_values, predictions))


def print_hyperparameters(hyperparameters):
    print(json.dumps(hyperparameters, sort_keys=True, indent=4))


def save_config(config: dict, file_name: str):
    if file_name is None:
        file_name = get_default_config_name()

    with open(f"hyperparameters/{file_name}.json", "w") as f:
        json.dump(config, f,
                  sort_keys=True,
                  indent=4)
    print(f"Saved config under {file_name}.json")


def save_predictions(predictions: "pd.DataFrame", score: float, file_name: str = None):
    if file_name is None:
        file_name = get_default_prediction_name(score)

    predictions.to_csv(f"results/{file_name}.csv", index=False)


def save_test_predictions(predictions: "pd.DataFrame", score: float, file_name: str = None):
    if file_name is None:
        file_name = get_default_prediction_name(score)

    predictions.to_csv(f"test_results/{file_name}.csv", index=False)


def get_default_config_name():
    return "config_" + time.strftime("%Y%m%d-%H%M%S")


def get_default_prediction_name(score: float):
    return f"{time.strftime('%Y%m%d-%H%M%S')}_s:{score}"


def save_checkpoint(current_config: dict, output_configs: dict):
    current_config = copy.deepcopy(current_config)
    current_config["config_list"].extend(output_configs)
    with open("checkpoints/checkpoint.json", "w") as f:
        json.dump(current_config, f, sort_keys=True, indent=4)
