import torch.nn as nn


class Parallel(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self._paths = nn.ModuleList(layers)

    def forward(self, x):
        return tuple(path(x) for path in self._paths)

    def string(self):
        return f"Parallel[{self._paths}]"
