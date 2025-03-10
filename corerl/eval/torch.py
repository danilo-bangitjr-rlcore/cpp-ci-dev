import torch


def stable_rank(matrix: torch.Tensor) -> float:
    r"""
    Calculate the stable rank of a matrix.

    Stable rank is defined as: \sum \sigma^2 / \max \sigma^2

    """
    singular_values = torch.linalg.svdvals(matrix)
    sv_squared = singular_values**2
    s_rank =  torch.sum(sv_squared) / sv_squared[0] # the max singular value is the first

    return s_rank.item()

def get_layers_stable_rank(model: torch.nn.Module) -> list[float]:
    """
    Iterate through model parameters and calculate stable rank for each weight matrix.
    """
    stable_ranks = []
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            stable_ranks.append(stable_rank(param))
    return stable_ranks
