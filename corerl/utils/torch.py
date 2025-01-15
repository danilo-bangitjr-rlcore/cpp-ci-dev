import torch


def tensor_allclose(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08):
    if t1.shape != t2.shape:
        return False
    return torch.allclose(t1, t2, rtol=rtol, atol=atol)
