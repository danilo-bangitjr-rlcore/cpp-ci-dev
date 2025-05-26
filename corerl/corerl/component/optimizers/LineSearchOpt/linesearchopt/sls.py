# pyright: basic
# Adapted from https://github.com/IssamLaradji/sls

import copy
import logging
from collections.abc import Iterable
from typing import Any, Callable, Generic, Optional, TypeVar
from warnings import warn

import torch

from corerl.state import AppState

from . import util as ut
from .init import StepsizeInit
from .search import Search

logger = logging.getLogger(__name__)

OPT = TypeVar("OPT", bound=torch.optim.Optimizer)  # pyright: ignore[reportPrivateImportUsage]


class Optimizer(torch.optim.Optimizer,Generic[OPT]):  # pyright: ignore[reportPrivateImportUsage]
    """Implements a stochastic line search.

    See `paper <https://arxiv.org/abs/1905.09997>`.
    """
    def __init__(
        self,
        app_state: AppState,
        params: Iterable[torch.nn.parameter.Parameter],
        optim: type[OPT],
        search_condition: Search,
        init: StepsizeInit,
        init_step_size: float,
        optim_args: tuple=tuple(),
        optim_kwargs: Optional[dict]=None,
        max_backtracking_steps: int=100,
        fallback_step_size: float=1e-6,
        unit_norm_direction: bool=False,
        reeval_loss_on_diff_start: bool=False,
    ):
        """Initializes the instance.

        Args:
            app_state:
                App state used for logging metrics.
            params:
                The parameters to optimise, usually gotten from a PyTorch model
                as `model.parameters()`
            optim:
                The PyTorch optimiser type to use in the line search. Do not
                pass in a constructed optimiser, instead, pass in the class
                itself.
            search_condition:
                The search condition to use, outlining how to update step sizes
            init: The method of re-initialising step sizes between line
                searches
            init_step_size:
                The step size to start the optimisation process at

            optim_args:
                Additional arguments passed to the optimiser
            optim_kwargs:
                Additional keyword arguments passed to the optimiser
            max_backtracking_steps:
                The maximum number of backtracking steps to use in the line
                search
            fallback_step_size:
                The value to use for the step size if the line search fails
            unit_norm_direction:
                Whether or not the search direction should
                have unit norm. This only adjusts the norm of the search
                direction in the line search conditions, but does not adjust
                the norm of the direction of the optimizer. For example, for
                the Armijo conditions:

                    f(x + αp₀) ≤ f(x) + αc ∇f(x)ᵀp₁                         (1)

                This is equivalent to adjusting `p₁` to have unit norm but
                **not** `p₀`. Hence, we can view this as a re-scaling of the
                `c` term. Note that this is not equivalent to a rescaling of
                the `c` term for the Goldstein conditions. Instead, it is
                equivalent to re-scaling both `c` and `(1-c)`.
            reeval_loss_on_diff_start: For certain optimizers (e.g. AdamW), the
                gradient step is taken from different parameters than the
                current network parameters. For example, in the case of
                `AdamW`, the parameters are first decayed before the gradient
                step is taken. The `reeval_loss_on_diff_start` parameter
                controls whether the loss on the updated network parameters
                should be compared to the loss on the original parameters or
                the loss on the parameters at which the gradient step is taken.
        """

        if optim_kwargs is None:
            optim_kwargs = {}

        # Never consider maximization
        assert (
            "maximize" not in optim_kwargs.keys() or
            not optim_kwargs["maximize"]
        )

        assert max_backtracking_steps > 0

        self._app_state = app_state
        params = list(params)
        self._optim_type = optim
        optimizer = optim(params, *optim_args, **optim_kwargs)
        defaults = dict(
            init=init,
            search_condition=search_condition,
            optim=optimizer,
            fallback_step_size=fallback_step_size,
            max_backtracking_steps=max_backtracking_steps,
            init_step_size=init_step_size,
            unit_norm_direction=unit_norm_direction,
            reeval_loss_on_diff_start=reeval_loss_on_diff_start,
        )
        super().__init__(params, defaults)

        self.state: dict[Any, Any]  # pyright: ignore[reportIncompatibleVariableOverride]
        self.state["step"] = 0
        self.state['step_size'] = init_step_size

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        def _state_dict_post_hook(opt, sd):
            sd["state"]["optim"] = []
            sd["state"]["optim_defaults"] = []
            for i, g in enumerate(self.param_groups):
                optim = g["optim"]
                sd["state"]["optim"].append(optim.state_dict()["state"])
                sd["state"]["optim_defaults"].append(optim.defaults)

                optim_params = optim.state_dict()["param_groups"]
                sd["param_groups"][i]["opt_param_group"] = optim_params

                del sd["param_groups"][i]["optim"]

            return sd

        self.register_state_dict_post_hook(_state_dict_post_hook)

    def add_param_group(self, param_group):
        if len(self.param_groups) > 0:
            warn(
                "Multiple param groups for Optimizer has not been tested",
                stacklevel=1
            )
        return super().add_param_group(param_group)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        for k, v in state_dict["state"].items():
            self.state[k] = v

        for i in range(len((self.param_groups))):
            opt_sd = {
                "state": state_dict["state"]["optim"][i],
                "param_groups": state_dict["param_groups"][i]["opt_param_group"],
            }

            self.param_groups[i]["optim"] = self._optim_type(
                self.param_groups[i]["params"],
                state_dict["state"]["optim_defaults"][i],
            )
            self.param_groups[i]["optim"].load_state_dict(opt_sd)

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        closure: Callable[[], float | torch.Tensor],
        loss: Optional[float]=None,
    ) -> float:
        """Takes a single optimization step to minimize `closure`.

        If `loss` is not `None`, then this function assumes that the objective
        at the current parameters of the model is `loss`. That is, `closure() =
        loss` at the current network parameters. This can improve efficiency by
        reducing the number of forward calls by 1.

        Args:
            closure:
                A closure which which serves as the objective function to
                minimize. The function should return a **scalar** value and
                take no parameters.
            loss:
                See description above.

        Returns:
            The objective function value for the previous and updated
            parameters respectively.
        """

        loss_next: Optional[float] = None  # makes static type-checkers happy

        batch_step_size = self.state['step_size']

        # Since we never need to compute gradients, we wrap the call to
        # closure() in a no_grad operation.
        def _closure():
            self.state['n_forwards'] += 1
            with torch.no_grad():
                return float(closure())

        # One backward pass was needed to get the gradient information and was
        # called by the user
        self.state['n_backwards'] += 1

        if loss is None:
            loss = _closure()

        for group in self.param_groups:
            params = group["params"]
            opt = group["optim"]
            # opt.zero_grad()

            unit_norm_direction = group["unit_norm_direction"]
            step_size = group["init"].reinit(batch_step_size)

            # Save the current parameters and optimiser state
            initial_params = copy.deepcopy(params)
            grad_current = ut._get_grad_list(params)
            opt_state_current = copy.deepcopy(opt.state_dict())

            with torch.no_grad():
                # Get the derivative in the search direction in order to run
                # the line search later
                directional_derivative = ut._compute_search_direction(
                    initial_params, grad_current, opt,
                    unit_norm_direction=unit_norm_direction,
                )

                # If the starting point for the line search
                reeval_loss_on_diff_start = group["reeval_loss_on_diff_start"]
                starting_params, changed = ut._compute_start_point(
                    initial_params, grad_current, opt,
                )

                # If the starting point of the line search is different from
                # the current parameters, update the original objective value
                # we consider in the line search if the user specified to do so
                if changed and reeval_loss_on_diff_start:
                    # Move parameters to starting position of line search
                    zipped = zip(params, starting_params, strict=True)
                    for p, p_start in zipped:
                        p.data[:] = p_start.data

                    # Re-evaluate loss without recording gradients
                    loss = _closure()

                    # Reset back to current network parameters
                    zipped = zip(params, initial_params, strict=True)
                    for p, p_init in zipped:
                        p.data[:] = p_init.data

                found = False
                # Run backtracking line search
                for backtrack_step in range(group["max_backtracking_steps"]):
                    # Backtracking step -> revert optimiser state to
                    # what it was before the last optimiser step
                    opt.load_state_dict(copy.deepcopy(opt_state_current))

                    # Backtracking step -> revert the parameter values to
                    # what they were before the last optimiser step
                    zipped = zip(params, initial_params, strict=True)
                    for p, p_init in zipped:
                        p.data[:] = p_init.data

                    for g in opt.param_groups:
                        g["lr"] = step_size

                    # breakpoint()
                    opt.step()

                    # compute the loss at the next step; no need to compute
                    # gradients.
                    loss_next = _closure()

                    found, step_size = group["search_condition"](
                        step_size, loss, directional_derivative, loss_next,
                    )
                    if found:
                        self._app_state.metrics.write(
                            agent_step=self._app_state.agent_step,
                            metric="optimizer_lso_step_size",
                            value=step_size,
                        )

                        self._app_state.metrics.write(
                            agent_step=self._app_state.agent_step,
                            metric="optimizer_lso_backtrack_steps",
                            value=backtrack_step,
                        )

                        break

                if not found:  # Line search exceeded maximum number of epochs
                    logger.warning("LSO max backtracking steps exceeded")
                    # Since we won't be using opt_state_current again,, it is
                    # okay to mutate it here
                    opt.load_state_dict(opt_state_current)

                    zipped = zip(params, initial_params, strict=False)
                    for p, p_init in zipped:
                        p.data[:] = p_init.data

                    for g in opt.param_groups:
                        g["lr"] = group["fallback_step_size"]

                    opt.step()

                    loss_next = _closure()

                    # Restart next line search at the fallback stepsize
                    step_size = group["fallback_step_size"]

                    self._app_state.metrics.write(
                        agent_step=self._app_state.agent_step,
                        metric="lso_step_size",
                        value=step_size,
                    )

                    self._app_state.metrics.write(
                            agent_step=self._app_state.agent_step,
                            metric="lso_backtrack_steps",
                            value=group["max_backtracking_steps"],
                    )

                group["init"].record_used(step_size)

            # save the new step-size
            self.state['step_size'] = step_size
            self.state['step'] += 1

        return loss
