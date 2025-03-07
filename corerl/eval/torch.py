import torch


def stable_rank(matrix: torch.Tensor) -> float:
    """
    Calculate the stable rank of a matrix.

    Stable rank is defined as: ||A||_F^2 / ||A||_2^2
    where ||A||_F is the Frobenius norm and ||A||_2 is the spectral norm.

    """
    frob_norm_squared = torch.sum(matrix ** 2)
    singular_values = torch.linalg.svdvals(matrix)
    spectral_norm_squared = singular_values[0] ** 2
    s_rank = frob_norm_squared / spectral_norm_squared

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
