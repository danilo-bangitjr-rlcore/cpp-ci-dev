# pyright: basic
from typing import Callable, Mapping

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import corerl.component.optimizers.LineSearchOpt.linesearchopt as lso

CLOSURE_CREATOR = Callable[
    [torch.nn.Module, int, torch.Tensor, torch.Tensor],
    Callable[[], torch.Tensor],
]


# ----------------------------------------------------------------
# Metadata for all tests
# ----------------------------------------------------------------

# The threshold to consider the network to have converged
underlying_threshold = 0.05
noisy_threshold = 0.05

# The seed for generating random numbers
seed = 1

# The min/max stepsizes to tuse for the Armijo line search
min_step_size = 0.0
max_step_size = 1.0

# The batch size and number of training steps to use in the training loop
batch_size = 32
learning_steps = 96


@pytest.fixture
def fn_dataset():
    def _fn_dataset(
        seed: int,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        n_points = 2500
        length = 2.
        noise = 0.1

        torch.manual_seed(seed)
        x = torch.rand(n_points).reshape(-1, 1) * length - length/2
        y = fn(x)

        return {
            "x": x, "y": y, "y_noise": y + torch.randn(n_points, 1) * noise,
        }

    return _fn_dataset


@pytest.fixture
def polynomial():
    def _polynomial(seed: int, degree: int) -> Mapping[str, torch.Tensor]:
        n_points = 2500
        length = 2.
        noise = 0.1

        torch.manual_seed(seed)
        x = torch.rand(n_points).reshape(-1, 1) * length - length/2
        y = torch.zeros(n_points, 1)

        for p in range(degree + 1):
            coeff = torch.rand(1).item() * 10 - 5
            y += coeff * (x ** p)

        return {"x": x, "y": y, "y_noise": y + torch.randn(n_points, 1) * noise}
    return _polynomial


@pytest.fixture
def net():
    def _net(seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)

        return nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    return _net


@pytest.fixture
def closure():
    def _closure(
        net: torch.nn.Module,
        batch_size: int,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        def _inner_closure():
            if batch_size < 1:
                return F.mse_loss(net(inputs), outputs)

            indices = torch.randint(inputs.shape[0], (batch_size,))
            x = inputs[indices, :]
            y = outputs[indices, :]
            return F.mse_loss(net(x), y)

        return _inner_closure
    return _closure


@pytest.fixture
def armijo_lso_no_backtrack():
    init = lso.init.Multiply(1.5)
    def _get_opt(
        net: torch.nn.Module,
        init_step_size: float=0.1,
        init: lso.init.StepsizeInit=init,
        c: float=0.1,
        beta: float=0.95,
    ) -> lso.Optimizer:
        condition = lso.search.Armijo(
            c=c, beta=beta, min_step_size=min_step_size,
            max_step_size=max_step_size,
        )
        return lso.Optimizer(
            net.parameters(),
            torch.optim.Adam,  #pyright: ignore[reportPrivateImportUsage]
            condition,
            init,
            init_step_size=init_step_size,
            unit_norm_direction=False,
            max_backtracking_steps=1
        )
    return _get_opt

@pytest.fixture
def armijo_lso_sgd():
    init = lso.init.Multiply(1.5)
    def _get_opt(
        net: torch.nn.Module,
        init_step_size: float=0.1,
        init: lso.init.StepsizeInit=init,
        c: float=0.1,
        beta: float=0.95,
    ) -> lso.Optimizer:
        condition = lso.search.Armijo(
            c=c, beta=beta, min_step_size=min_step_size,
            max_step_size=max_step_size,
        )
        return lso.Optimizer(
            net.parameters(),
            torch.optim.SGD,  #pyright: ignore[reportPrivateImportUsage]
            condition,
            init,
            init_step_size=init_step_size,
            unit_norm_direction=False,
        )
    return _get_opt

@pytest.fixture
def armijo_lso():
    init = lso.init.Multiply(1.5)
    def _get_opt(
        net: torch.nn.Module,
        init_step_size: float=0.1,
        init: lso.init.StepsizeInit=init,
        c: float=0.1,
        beta: float=0.95,
    ) -> lso.Optimizer:
        condition = lso.search.Armijo(
            c=c, beta=beta, min_step_size=min_step_size,
            max_step_size=max_step_size,
        )
        return lso.Optimizer(
            net.parameters(),
            torch.optim.Adam,  #pyright: ignore[reportPrivateImportUsage]
            condition,
            init,
            init_step_size=init_step_size,
            unit_norm_direction=False,
        )
    return _get_opt


def train_test_loop(
    net: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    closure: CLOSURE_CREATOR,
    opt: lso.Optimizer,
    use_loss_kwarg,
):
    _closure = closure(net, batch_size, data["x"], data["y_noise"])
    for _ in range(learning_steps):
        train_step(opt, _closure, use_loss_kwarg)

    loss_on_underlying = closure(net, 0, data["x"], data["y"])()
    loss_on_noisy = closure(net, 0, data["x"], data["y_noise"])()

    return (
        loss_on_underlying < underlying_threshold and
        loss_on_noisy < noisy_threshold
    )


def train_step(
    opt: lso.Optimizer,
    closure: Callable[[], torch.Tensor],
    use_loss_kwarg: bool,
):
    loss = closure()
    opt.zero_grad()
    loss.backward()
    if use_loss_kwarg:
        opt.step(closure, loss=float(loss))
    else:
        opt.step(closure)


def test_linear(net, polynomial, closure, armijo_lso):
    degree = 1
    net = net(seed)
    data = polynomial(seed, degree)
    opt = armijo_lso(net, init=lso.init.To(0.1))
    assert train_test_loop(net, data, closure, opt, use_loss_kwarg=False)

def test_linear_no_backtrack(net, polynomial, closure, armijo_lso_no_backtrack):
    """
    Show that base fixed learning rate fails to solve the problem
    """
    degree = 1
    net = net(seed)
    data = polynomial(seed, degree)
    opt = armijo_lso_no_backtrack(net, init=lso.init.To(0.1))
    assert not train_test_loop(net, data, closure, opt, use_loss_kwarg=False)

def test_linear_sgd(net, polynomial, closure, armijo_lso_sgd):
    degree = 1
    net = net(seed)
    data = polynomial(seed, degree)
    opt = armijo_lso_sgd(net, init=lso.init.To(0.1))
    assert train_test_loop(net, data, closure, opt, use_loss_kwarg=False)

def test_quintic(net, polynomial, closure, armijo_lso):
    degree = 5
    net = net(seed)
    data = polynomial(seed, degree)
    opt = armijo_lso(net, init=lso.init.To(0.1))
    assert train_test_loop(net, data, closure, opt, use_loss_kwarg=False)


def test_cos(net, fn_dataset, closure, armijo_lso):
    net = net(seed)
    data = fn_dataset(seed, torch.cos)
    opt = armijo_lso(net, init=lso.init.To(0.1))
    assert train_test_loop(net, data, closure, opt, use_loss_kwarg=False)


def test_loss_kwarg(net, fn_dataset, closure, armijo_lso):
    net = net(seed)
    data = fn_dataset(seed, torch.cos)
    opt = armijo_lso(net, init=lso.init.To(0.1))
    assert train_test_loop(net, data, closure, opt, use_loss_kwarg=True)
