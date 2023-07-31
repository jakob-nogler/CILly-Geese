import typing
from .model import Model
from .autoencoder.model import Autoencoder
from .ncf.model import NCF
from .als.model import ALSModel
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
    MODEL_CLASSES = {
        'autoencoder': Autoencoder,
        'ncf': NCF,
        'als': ALSModel,
        'lightgcn': LightGCN,
        'funksvd': funkSVD,
        'svdplusplus': SVDplusplus,
        'oneidea': CFDA,
        'cfda': CFDA,
        'gernot': Gernot,
        'bayesiansvd': BayesianSVD,
    }

    model_class = MODEL_CLASSES[name.lower()]
    return model_class(hyperparameters, train_set, test_set)
