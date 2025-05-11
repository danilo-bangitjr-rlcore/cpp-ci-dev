import torch


def tensor_allclose(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08):
    if t1.shape != t2.shape:
        return False
    return torch.allclose(t1, t2, rtol=rtol, atol=atol)


def clip_gradients(module: torch.nn.Module, max_norm: float):
    original_grads = [
        torch.clone(p.grad).detach()
        if p.grad is not None
        else None
        for p in module.parameters()
    ]
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

    grad_deltas = [
        torch.abs(torch.sub(og, clipped.grad)).max().item()
        for og, clipped in zip(original_grads, module.parameters(), strict=True)
        if og is not None and clipped.grad is not None
    ]

    return float(max(grad_deltas))
