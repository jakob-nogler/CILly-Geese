import typing
from .model import Model
from .autoencoder.model import Autoencoder
from .ncf.model import NCF
from .als.model import ALSModel
from .vae.model import VAE
from .LightGCN.model import LightGCN
from .funkSVD.model import funkSVD
from .SVDplusplus.model import SVDplusplus
from .cfda.model import CFDA
from .gernot.model import Gernot
from .BayesianSVD.model import BayesianSVD

if typing.TYPE_CHECKING:
    import pandas as pd


def get_model(name: str, hyperparameters: dict, train_set: "pd.DataFrame", test_set: "pd.DataFrame") -> Model:
    """Returns a model based on the name string -> initialized with the specified parameters
    """
    if name == "autoencoder":
        return Autoencoder(hyperparameters, train_set, test_set)
    elif name == "ncf":
        return NCF(hyperparameters, train_set, test_set)
    elif name == "als":
        return ALSModel(hyperparameters, train_set, test_set)
    elif name == "vae":
        return VAE(hyperparameters, train_set, test_set)
    elif name == "lightgcn":
        return LightGCN(hyperparameters, train_set, test_set)
    elif name == "funkSVD":
        return funkSVD(hyperparameters, train_set, test_set)
    elif name == "SVDplusplus":
        return SVDplusplus(hyperparameters, train_set, test_set)
    elif name == "oneidea" or name == "cfda":
        return CFDA(hyperparameters, train_set, test_set)
    elif name == "gernot":
        return Gernot(hyperparameters, train_set, test_set)
    elif name == "bayesiansvd":
        return BayesianSVD(hyperparameters, train_set, test_set)
    return None
