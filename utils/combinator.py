import torch.nn as nn
import torch


class Combinator(nn.Module):
    """Module performing an affine combination of the input vectors with trainable weights
    """

    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.num_models = num_models
        self.weights = nn.Parameter(torch.ones(num_models))

    def forward(self, predictions: "torch.Tensor"):
        w = torch.softmax(self.weights, 0).float()
        return torch.mv(torch.transpose(predictions.float(), 0, 1), w)

    def set_weights(self, weights: "torch.Tensor"):
        self.weights = nn.Parameter(weights)

    def get_weights(self) -> "torch.Tensor":
        return self.weights
