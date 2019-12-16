from copy import deepcopy
from torch import nn


def clones(layer, n):
    """Produces several copies of a layer.

    Args:
        layer (torch.nn.Module): any layer.
        n (int): clone times.

    Returns:
        layers (torch.nn.ModuleList): stack of layers.
    """
    return nn.ModuleList([deepcopy(layer) for _ in range(n)])
