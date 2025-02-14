import logging
from abc import ABC, abstractmethod

import torch
from numpy import clip as clip
from typing_extensions import override

logger = logging.getLogger(__name__)

class Search(ABC):
    """Implements a line search for stepsizes.
    """
    def __init__(self, min_step_size: float, max_step_size: float):
        self._min_step_size = min_step_size
        self._max_step_size = max_step_size

    def __call__(
        self,
        step_size: float,
        loss: float,
        grad_norm: float,
        loss_next: float,
    ):
        return self.step(step_size, loss, grad_norm, loss_next)

    def step(
        self,
        step_size: float,
        loss: float,
        directional_derivative: float,
        loss_next: float,
    ) -> tuple[bool, float]:
        """Adjusts the argument stepsize based on the search condition.

        This function checks the search conditions and adjusts the argument
        stepsize accordingly. The function also clips stepsizes to be within
        the minimum and maximum allowable range, as defined by the inputs to
        the `Search` constructor.

        Returns:
            Whether the search conditions were satisfied by `stepsize` and the
            next stepsize to use in the line search/optimization step.
        """
        found, step_size = self._step(
            step_size, loss, directional_derivative, loss_next,
        )
        # breakpoint()
        return found, clip(step_size, self._min_step_size, self._max_step_size)

    @abstractmethod
    def _step(
        self,
        step_size: float,
        loss: float,
        directional_derivative: float,
        loss_next: float,
    ) -> tuple[bool, float]:
        """Checks the search conditions and adjusts the stepsize accordingly.
        """
        pass


class Armijo(Search):
    """Implements the Armijo conditions for a stepsize line search.

    Armijo implements the Armijo line search algorithm. A stepsize `α` is
    accepted based on the Armijo condition:

        f(x + αp) ≤ f(x) + αc ∇f(x)ᵀp                                 (1)

    where `f` is the objective function, `p` is the search direction
    (determined by the optimizer), x is the parameters to be optimized, `m =
    ∇f(x)ᵀp` is the derivative in the search direction (determined by the
    optimizer), and `c` is a hyperparameter which determines the threshold for
    the line search. If the equation above is not satisfied, then the stepsize
    is updated as

        α ← α * β

    for `β ∈ (0, 1)`

    Args:
        c:
            The `c` parameter in Equation (1)
        beta:
            The multiplicative factor to decrease the stepsize with, in (0, 1)
        min_step_size:
            The minimum allowable stepsize to use
    """
    def __init__(
        self,
        c: float,
        beta: float,
        min_step_size: float=0,
        max_step_size: float=10,
    ):
        self._c = c
        assert 0 < beta < 1
        self._beta = beta
        super().__init__(min_step_size, max_step_size)

    @override
    def _step(
        self,
        step_size: float,
        loss: float,
        directional_derivative: float,
        loss_next: float,
    ) -> tuple[bool, float]:
        thresh = (loss + step_size * self._c * directional_derivative)
        break_condition = loss_next - thresh
        if break_condition <= 0:
            return True, step_size
        else:
            # Decrease the step-size by a multiplicative factor
            return False, step_size * self._beta


class Goldstein(Search):
    """
    Goldstein implements the ArmijoGoldstein line search algorithm. A stepsize
    `α` is accepted based on the ArmijoGoldstein conditions:

        f(x + αp) ≤ f(x) + αc ∇f(x)ᵀp                                       (2)
        f(x + αp) >= f(x) + α(1-c) ∇f(x)ᵀp                                   (3)

    where `f` is the objective function, `p` is the search direction
    (determined by the optimizer), x is the parameters to be optimized, `m =
    ∇f(x)ᵀp` is the derivative in the search direction (determined by the
    optimizer), and `c` is a hyperparameter which determines the threshold for
    the line search. If only Equation (3) above is satisfied, then the stepsize
    is updated as

        α ← α * β_b

    for `β_b ∈ (0, 1)`. Otherwise if only Equation (2) above is satisfied, then
    the stepsize is updates as:

        α ← α * β_f

    for `β_f > 1`.

    Parameters
    ----------
    c : float
        The `c` parameter in Equation (1)
    beta_b : float
        The multiplicative factor to decrease the stepsize with, in (0, 1)
    beta_f : float, default 2
        The multiplicative factor to increase the stepsize with, > 1
    min_step_size : float, default 0
        The minimum allowable stepsize to use
    max_step_size : float, default 10
        The maximum allowable stepsize to use
    """
    def __init__(
        self,
        c: float,
        beta_b: float,
        beta_f: float=2.0,
        min_step_size: float=0,
        max_step_size: float=10,
    ):
        self._c = c
        assert 0 < beta_b < 1
        self._beta_b = beta_b
        assert beta_f > 1
        self._beta_f = beta_f
        super().__init__(min_step_size, max_step_size)

    @override
    def _step(
        self,
        step_size: float,
        loss: float,
        directional_derivative: float,
        loss_next: float,
    ) -> tuple[bool, float]:
        found = 0

        if(
            loss_next <= (
                loss + (step_size) * self._c * directional_derivative
            )
        ):
            found = 1

        if(
            loss_next >= (
                loss + (step_size) * (1 - self._c) * directional_derivative
            )
        ):
            if found == 1:
                found = 3 # Both conditions are satisfied
            else:
                found = 2 # Only the curvature condition is satisfied

        if (found == 0):
            raise ValueError(
                'Error, neither Goldstein condition was satisfied'
            )
        elif (found == 1):
            # Step-size might be too small
            step_size = step_size * self._beta_f
        elif (found == 2):
            # Step-size might be too large
            step_size = step_size * self._beta_b

        return found == 3, step_size

torch.serialization.add_safe_globals([Armijo, Goldstein])
