from models.model import Model
from models.get_model import get_model
from utils.early_stopping import EarlyStopping
import random
import copy
import typing
import hashlib
import json
import os
import wandb


if typing.TYPE_CHECKING:
    import pandas as pd


def get_trained_model(config: dict, train_set: "pd.DataFrame", test_set: "pd.DataFrame", verbose: bool = False, save_model: bool = True, use_wandb: bool = False, full_data: bool = False) -> "typing.Tuple[Model, dict, list[dict]]":
    """Trains a model and returns it based on the config file

    Args:
        config (dict): config of the model to be trained
        train_set (pd.DataFrame): training set
        test_set (pd.DataFrame): validation set
        verbose (bool, optional): Print detailed training information to console. Defaults to False.
        save_model (bool, optional): Save the trained models to files. Defaults to True.
        use_wandb (bool, optional): Use the wandb framework. Defaults to False.

    Returns:
        typing.Tuple[Model, dict, list[dict]]: Returns a tuple (trained_model, model_hyperparameters, training_log)
    """

    if not use_wandb:
        # randomly draw hyperparameters from the lists as specified in the config file
        hyperparameters = draw_hyperparameters(config["hyperparameters"])
    else:
        # draw hyperparameters according to the wandb framework
        wandb.init()
        hyperparameters = vars(wandb.config)['_items']
        if 'seed' not in hyperparameters:
            hyperparameters['seed'] = random.randrange(1, 1e6)

    # initialize the model
    model = get_model(config["model"], hyperparameters, train_set, test_set)

    # check if the model has already been saved -> use the already trained model if so
    model_hash = get_model_hash(config["model"], hyperparameters, full_data)
    directory = f"./trained_models/{config['model']}"
    path = f"./trained_models/{config['model']}/{model_hash}"

    if os.path.exists(path):
        print(f"Found trained {config['model']} model, loading...")
        model.load(path)
        return model, hyperparameters, {}

    # train model
    print(f"Training {config['model']}...")
    log, min_epoch = train_model(model,
                                 verbose=verbose,
                                 use_wandb=use_wandb,
                                 full_data=full_data
                                 )

    hyperparameters["num_epochs"] = min_epoch

    # save the model to file
    if save_model:
        os.makedirs(directory, exist_ok=True)
        model.save(path)

    return model, hyperparameters, log


def train_model(model: Model, verbose: bool = False, use_wandb: bool = False, full_data: bool = False) -> "list[dict]":
    # trains the model

    log = []

    if "patience" not in model.hyperparameters:
        model.hyperparameters["patience"] = 0

    stopper = EarlyStopping(
        model.hyperparameters["patience"] * 2 + 2, delta=0.000001, verbose=verbose)

    i = 1

    max_epochs = None

    if "num_epochs" in model.hyperparameters:
        max_epochs = model.hyperparameters["num_epochs"]

    min_score = 50000
    min_epoch = -1

    # train the model as long as there is progress on the validation set
    while (not max_epochs and not stopper.early_stop) or (max_epochs and i <= max_epochs):
        # one epoch
        epoch_log = {}

        loss = model.train()
        epoch_log["loss"] = loss

        # compute the validation score and pass it to the early stopper
        test_score = model.test()
        stopper.step(test_score)
        epoch_log["test_score"] = test_score

        if test_score <= min_score:
            min_epoch = i

        if verbose:
            print(
                f"Epoch {i} | loss: {loss} | test score: {test_score}")

        log.append(epoch_log)

        # log the epoch to the wandb server
        if use_wandb:
            wandb.log({
                'epoch': i,
                'validation_score': test_score,
                'loss': loss
            })

        i += 1

    return log, min_epoch


def draw_hyperparameters(hyperparameters: dict) -> dict:
    """Draws parameters uniformly at random from as specified in the config file

    Args:
        hyperparameters (dict): _description_

    Returns:
        dict: _description_
    """

    hyperparameters = copy.deepcopy(hyperparameters)

    for key in hyperparameters:
        if type(hyperparameters[key]) is list:
            i = random.randint(0, len(hyperparameters[key])-1)
            hyperparameters[key] = hyperparameters[key][i]

    if 'seed' not in hyperparameters or not hyperparameters['seed']:
        hyperparameters['seed'] = random.randrange(1, 1e6)
    return hyperparameters


def get_model_hash(name: str, hyperparameters: dict, full_data: bool):
    """Generates a hash string based on the model hyperparameters

    Args:
        name (str): name of the model used
        hyperparameters (dict): model hyperparameters

    Returns:
        _type_: hash string
    """
    dictionary = {
        "name": name,
        "hyperparameters": hyperparameters,
        "full_data": full_data
    }
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
