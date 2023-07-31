# CIL FS23 Collaborative Filtering project

This repository contains the code for the Collaborative Filtering for the Computational Intelligence Lab FS23 by the team "CILly Geese" (Carmel Baharav, Alessandro Girardi, Patryk Morawski, Jakob Nogler).
It contains implementation of various state-of-the-art collaborative filtering models (e.g. CFDA, NCF, BayesianSVD++) as well as two new architectures that were developed by us (CFDA++ and GERNOT).
All of it is wrapped into a unified training framework which allows running hyperparameter search for each of the model and combining the results in an ensemble of the models in a simple manner.
It is also allows for integration with [wandb](https://wandb.ai/).

# Installation

Before running the project, please make sure that python 3.9 is installed on your machine, you can use conda or venv to set up the requirements.
To use conda, run the following:

```sh
conda create -n cilly-fs23 python==3.9
conda activate cilly-fs23
# Run this in the same folder as this README
python -m pip install -r requirements.txt
```

# Quick start

To train the ensemble used for the final submission please run

`python main.py --full_data`.

After finishing, the final predictions to submit to Kaggle will appear under the `results/` folder under a name containing the current date.

To run a single model please run

`python main.py --config model-name.json [--full_data]`,

where $\text{model-name } \in \{\text{als, autoencoder, bayesiansvd, funksvd, gernot, lightgcn, ncf, cfda} \}$ is the name of the model you wish to run.
Please use the `--full_data` flag in case you want to train the model on the full dataset (to generate a Kaggle submission) and leave it out in case you want to obtain the local validation score.

# Implemented Models

For an extensive description of the implemented models and their performance please refer to our report.

| Model           | Internal name | Used library                                               | Citations                                                                                                                                                                    |
| --------------- | ------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ALS             | `als`         | Own implementation                                         | [1](https://doi.org/10.1109/MC.2009.263)                                                                                                                                     |
| funkSVD         | `funkSVD`     | [funk-svd](https://github.com/gbolmier/funk-svd)           | [1](https://doi.org/10.1109/MC.2009.263), [2](https://sifter.org/simon/journal/20061211.html)                                                                                |
| SVD++           | `SVDplusplus` | [surprise](https://surpriselib.com/)                       | [1](https://doi.org/10.1109/MC.2009.263), [2](https://doi.org/10.1145/1401890.1401944)                                                                                       |
| Bayesian SVD ++ | `bayesiansvd` | [myfm](https://myfm.readthedocs.io/en/latest/)             | [1](https://doi.org/10.1145/1401890.1401944), [2](https://doi.org/10.1145/1390156.1390267), [3](https://doi.org/10.1109/ICDM.2010.127), [4](http://arxiv.org/abs/1905.01395) |
| NCF             | `ncf`         | Own implementation                                         | [1](https://doi.org/10.1145/3038912.3052569), [2](https://doi.org/10.1145/3383313.3412488)                                                                                   |
| Autoencoder     | `autoencoder` | Own implementation                                         | [1](https://doi.org/10.1145/2740908.2742726), [2](https://inria.hal.science/hal-01256422v1/document), [3](http://arxiv.org/abs/1708.01715)                                   |
| CFDA/CFDA++     | `cfda`        | Own implementation                                         | [1](https://doi.org/10.1109/IJCNN55064.2022.9892705)                                                                                                                         |
| LightGCN        | `lightgcn`    | Adapted from the [paper](https://arxiv.org/abs/2002.02126) | [1](https://arxiv.org/abs/2002.02126), [2](http://arxiv.org/abs/1905.08108)                                                                                                  |
| GERNOT          | `gernot`      | Own implementation                                         |

# Running the framework

You can start the project by running `python main.py` with the following optional arguments:

| Argument              | Explanation                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `--config FILE_NAME`  | use the config file under `hyperparameters/FILE_NAME`                                                       |
| `--verbose`           | Print detailed information during the training process                                                      |
| `--full_data`         | Use the full train set to train the models. Otherwise 90% of it is used for training and 10% for validation |
| `--save_models`       | Save the trained models under `trained_models/`                                                             |
| `--wandb`             | Use the wandb framework for hyperparameter search. Sweep configuration must be defined in the config file   |
| `--sweep_id SWEEP_ID` | Restart an existing wandb sweep with a given id                                                             |

The program will train the models as specified in the config file and construct an ensemble of the predictions of each of them.
The final prediction, which can be submitted to Kaggle, will be save under the `results/` folder.

### Config files

To set up which models and with which hyperparameters should be trained, you can edit or create a config file under the `hyperparameters/` folder.
They should be in the `.json` format and have the following structure:

| Parameter name        | Explanation                                                       |
| --------------------- | ----------------------------------------------------------------- |
| `config_list`         | List with configurations of the single models                     |
| `combination_weights` | Weights for the ensemble. If not specified, they will be trained. |

Each of the configurations in the config list should have the following structure:

| Parameter name        | Explanation                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `baseline`            | Minimal validation score the model has to achieve to be included in the ensemble                                                                                                                                                                                                                                                                                                                                         |
| `num_repetitions`     | The number of times a model with these hyperparameters will be trained.                                                                                                                                                                                                                                                                                                                                                  |
| `model`               | The name of the model to be trained. This should be one of: `autoencoder`, `ncf`, `lightgcn`, `als`, `funkSVD`, `SVDplusplus`, `cfda`, `gernot`, `bayesiansvd`                                                                                                                                                                                                                                                           |
| `hyperparameters`     | An object specifying the hyperparameters for the model. Which hyperparameters need to be specified, depends on the model - please consult the example config files for each model under `hyperparameters/` to get a more extensive overview. Instead of specifying a single value for a given parameter, a list with different options can also be used. In that case, one value will be chosen from the list at random. |
| `sweep_configuration` | Configuration of the wandb sweep. Please consult the wandb [documentation](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) to learn more about a sweep configuration structure.                                                                                                                                                                                                                          |

# Project Structure

| File or folder        | Explanation                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`             | Starting file for the project. Reads and processes arguments from the terminal.                                                       |
| `training_process.py` | Implementation of the training process which uses the specified config file to train the corresponding models and create an ensemble. |
| `train_model.py`      | A skeleton code for training a model                                                                                                  |
| `models/`             | Implementations of the various models                                                                                                 |
| `hyperparameters/`    | Config files                                                                                                                          |
| `data/`               | Contains the full train dataset as well as a split into training and validation sets.                                                 |
| `results/`            | Final predictions to submit to Kaggle                                                                                                 |
