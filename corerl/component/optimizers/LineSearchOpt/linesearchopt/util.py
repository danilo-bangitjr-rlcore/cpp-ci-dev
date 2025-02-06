# pyright: basic
# Adapted from https://github.com/IssamLaradji/sls

import copy
import warnings
from collections.abc import Iterable, Sequence
from typing import Union

import torch
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD

starting_point_unchanged_opts = set((Adam, SGD, RMSprop, Adagrad))


def _compute_start_point(param_list, grad_list, opt):
    """
    Computes the starting point for the line search for the optimizer `opt`
    using the network parameters `param_list` with gradients `grad_list`.

    If the optimizer is unknown, then this function prints a warning message
    and use the current parameters as an approximate starting point.

    Returns
    -------
    float, bool
        The starting point of the search direction for the optimizer `opt` and
        a boolean indicating whether or not the starting point for the line
        search has changed.
    """
    if type(opt) in starting_point_unchanged_opts:
        return param_list, False
    elif isinstance(opt, AdamW):
        return _compute_adamw_start_point(param_list, grad_list, opt), True
    else:
        warnings.warn(
            f"unknown start point for type {type(opt)}, approximating " +
            "start point with current parameters",
            stacklevel=1,
        )
        return param_list

def _compute_adamw_start_point(param_list, grad_list, opt):

    param_group = opt.param_groups
    assert len(param_group) == 1  # One param group per optimizer
    param_group = param_group[0]

    λ = param_group["weight_decay"]
    γ = param_group["lr"]

    return [p - (γ * λ * p) for p in param_list]


def _compute_search_direction(
    param_list: Sequence[torch.Tensor],
    grad_list: Sequence[torch.Tensor | None],
    opt: torch.optim.Optimizer,  #pyright: ignore[reportPrivateImportUsage]
    *,
    unit_norm_direction: bool,
):
    """
    Computes the search direction for the optimizer `opt` using the network
    parameters `param_list` with gradients `grad_list`.

    If `unit_norm_direction`, then return a directional derivative where the
    direction has unit norm.

    If the optimizer is unknown, then this function prints a warning message
    and uses the gradient direction as an approximate search direction.

    Returns
    -------
    float
        The derivative in the search direction for the optimizer `opt`.
    """
    if isinstance(opt, SGD):
        return _compute_sgd_direction(
            param_list, grad_list, opt, unit_norm_direction,
        )
    elif isinstance(opt, RMSprop):
        return _compute_rmsprop_direction(
            param_list, grad_list, opt, unit_norm_direction,
        )
    elif isinstance(opt, Union[Adam, AdamW]):
        return _compute_adam_direction(
            param_list, grad_list, opt, unit_norm_direction,
        )
    else:
        warnings.warn(
            f"unknown search direction for type {type(opt)}, approximating " +
            "search direction with gradient direction",
            stacklevel=1
        )
        return -_compute_grad_norm2(grad_list)


def _compute_sgd_direction(
    param_list: Sequence[torch.Tensor],
    grad_list: Sequence[torch.Tensor | None],
    opt: SGD,
    unit_norm_direction: bool=False,
) -> torch.Tensor:
    if len(opt.state_dict()["state"].keys()) == 0:
        return -_compute_grad_norm2(grad_list)
    else:
        directional_derivative = torch.tensor(0.)

        # SGD hypers
        param_group = opt.param_groups
        assert len(param_group) == 1  # One param group per optimizer
        param_group = param_group[0]
        λ = param_group["weight_decay"]
        μ = param_group["momentum"]
        τ = param_group["dampening"]
        nesterov = param_group["nesterov"]

        if λ == 0 and μ == 0:
            return -_compute_grad_norm2(grad_list)

        for i, ind in enumerate(opt.state_dict()["state"].keys()):
            g = copy.deepcopy(grad_list[i])
            p = param_list[i]

            if g is None:
                continue

            if λ != 0:
                g += (λ * p)

            if μ != 0:
                momentum = opt.state_dict()["state"][ind]["momentum_buffer"]
                if momentum is not None:  # After first step
                    b = μ * momentum + (1 - τ) * g
                else:
                    b = g

                if nesterov:
                    g += (μ * b)
                else:
                    g = b

            # Ensure unit-norm direction
            if unit_norm_direction:
                direction = g / torch.linalg.vector_norm(g)
            else:
                direction = g

            directional_derivative += torch.sum(torch.mul(g, direction))

    return -directional_derivative


def _compute_rmsprop_direction(
    param_list: Sequence[torch.Tensor],
    grad_list: Sequence[torch.Tensor | None],
    opt: RMSprop,
    unit_norm_direction: bool=False,
) -> torch.Tensor:
    if len(opt.state_dict()["state"].keys()) == 0:
        return -_compute_grad_norm2(grad_list)
    else:
        directional_derivative = torch.tensor(0.)

        # RMSprop hypers
        param_group = opt.param_groups
        assert len(param_group) == 1  # One param group per optimizer
        param_group = param_group[0]
        ε = param_group["eps"]
        λ = param_group["weight_decay"]
        α = param_group["alpha"]
        μ = param_group["momentum"]
        centered = param_group["centered"]

        for i, ind in enumerate(opt.state_dict()["state"].keys()):
            g = copy.deepcopy(grad_list[i])
            p = param_list[i]

            if g is None:
                continue

            if λ != 0:
                g += (λ * p)

            v = (
                α * opt.state_dict()["state"][ind]["square_avg"] +
                (1 - α) * (g ** 2)
            )

            if centered:
                g_avg = (
                    α * opt.state_dict()["state"][ind]["grad_avg"] +
                    (1 - α) * g
                )
                v -= (g_avg ** 2)

            if μ > 0:
                direction = (
                    μ * opt.state_dict()["state"][ind]["momentum_buffer"] +
                    (g / (torch.sqrt(v) + ε))
                )
            else:
                direction = g / (torch.sqrt(v) + ε)

            # Ensure unit-norm direction
            if unit_norm_direction:
                direction /= torch.linalg.vector_norm(direction)

            directional_derivative += torch.sum(torch.mul(g, direction))

    return -directional_derivative


def _compute_adam_direction(
    param_list: Sequence[torch.Tensor],
    grad_list: Sequence[torch.Tensor | None],
    opt: Union[Adam, AdamW],
    unit_norm_direction: bool=False,
    approximate_grad: bool=False
) -> torch.Tensor:
    if len(opt.state_dict()["state"].keys()) == 0:
        return -_compute_grad_norm2(grad_list)
    else:
        directional_derivative = torch.tensor(0.)

        # Adam hypers
        param_group = opt.param_groups
        assert len(param_group) == 1  # One param group per optimizer
        param_group = param_group[0]
        eps = param_group["eps"]
        β1, β2 = param_group["betas"]
        ams_grad = param_group["amsgrad"]
        λ = param_group["weight_decay"]

        for i, ind in enumerate(opt.state_dict()["state"].keys()):
            g = copy.deepcopy(grad_list[i])
            p = param_list[i]
            step = opt.state_dict()["state"][ind]["step"]

            if g is None:
                continue

            if λ != 0:
                g += (λ * p)

            m = (
                opt.state_dict()["state"][ind]["exp_avg"] * β1 +
                (1 - β1) * g
            )
            mhat = m / (1 - β1 ** step)

            v = (
                opt.state_dict()["state"][ind]["exp_avg_sq"] * β2 +
                (1 - β2) * (g ** 2)
            )
            vhat = v / (1 - β2 ** step)

            if ams_grad:
                vhat_max = opt.state_dict()["state"][ind]["max_exp_avg_sq"]
                vhat = torch.maximum(vhat, vhat_max)

            direction = mhat / (torch.sqrt(vhat) + eps)

            # Ensure unit-norm direction
            if unit_norm_direction:
                direction /= torch.linalg.vector_norm(direction)

            if approximate_grad:
                # treat p as an approximation of ∇f(x), implements condition:
                # f(x + αp) ≤ f(x) + αc pᵀp                                 (1)
                directional_derivative += torch.sum(torch.mul(direction * torch.linalg.vector_norm(g), direction))

            else:
                # use ∇f(x) for the linear approximation instead of p
                # f(x + αp) ≤ f(x) + αc ∇f(x)ᵀp                            (2)
                directional_derivative += torch.sum(torch.mul(g, direction))

    return -directional_derivative


def _compute_grad_norm(grad_list: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(_compute_grad_norm2(grad_list))


def _compute_grad_norm2(
    grad_list: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    grad_norm = torch.tensor(0.)
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    return grad_norm


def _get_grad_list(
    params: Iterable[torch.Tensor],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(p.grad for p in params)
