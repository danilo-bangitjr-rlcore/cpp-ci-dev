import torch
from torch import nn


class Parallel(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self._paths = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        return tuple(path(x) for path in self._paths)
