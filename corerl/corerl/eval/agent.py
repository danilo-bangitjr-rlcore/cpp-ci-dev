from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec, Protocol, SupportsFloat, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch
from lib_agent.buffer.buffer import State
from pydantic import BaseModel

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import (
    GreedyAC,
)
from corerl.agent.utils import get_sampled_qs
from corerl.component.network.utils import tensor
from corerl.configs.config import config
from corerl.state import AppState
from corerl.utils.device import device

# --------------------------------- Utilities --------------------------------- #

class EvalConfig(Protocol):
    enabled: bool

T = TypeVar('T', bound=EvalConfig)
P = ParamSpec('P')
EvalFn = Callable[Concatenate[T, GreedyAC, P], Sequence[torch.Tensor]]

def agent_eval(
    app_state: AppState,
    agent: BaseAgent,
    cfg_lens : Callable[[AppState], T],
    eval_fn: EvalFn[T, P],
    metric_names: list[str],
    *args: P.args,
    **kwargs: P.kwargs,
):
    cfg = cfg_lens(app_state)
    if not cfg.enabled:
        return

    if not isinstance(agent, GreedyAC):
        return

    metrics_tuple = eval_fn(cfg, agent, *args, **kwargs)

    if len(metrics_tuple) == 0:
        return

    for name, values in zip(metric_names, metrics_tuple, strict=True):
        metrics: dict[str, SupportsFloat] = {}
        if values.numel() == 1:
            metrics[name] = values.max()
        else:
            metrics = {
                f'{name}_max':  values.max(),
                f'{name}_min':  values.min(),
                f'{name}_mean': values.mean(),
                f'{name}_var':  values.var(),
            }

        for k, v in metrics.items():
            app_state.metrics.write(app_state.agent_step, k, v)

SAEvalFn = Callable[[T, GreedyAC, torch.Tensor, torch.Tensor, torch.Tensor], Sequence[torch.Tensor]]
BatchSAEvalFn = Callable[[EvalConfig, GreedyAC], Sequence[torch.Tensor]]
def policy_buffer_batchify(eval_fn: SAEvalFn) ->  BatchSAEvalFn:
    def batchified(cfg: EvalConfig, agent: GreedyAC):
        if not agent.is_policy_buffer_sampleable:
            return ()

        batches = agent.sample_policy_buffer()
        assert len(batches) == 1
        batch = batches[0]
        state_batch = batch.prior.state
        action_lo_batch = batch.prior.action_lo
        action_hi_batch = batch.prior.action_hi

        return eval_fn(cfg, agent, state_batch, action_lo_batch, action_hi_batch)

    return batchified


def ensure_2d_tensor(array : np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        array = tensor(array, device=device.device)
    if array.dim() == 1:
        array = array.unsqueeze(0)

    assert array.dim() == 2
    return array


# ------------------------------ Policy Variance ----------------------------- #

@config()
class PolicyVarianceConfig:
    enabled: bool = True
    n_samples: int = 100

def _policy_variance(
    cfg: PolicyVarianceConfig,
    agent: GreedyAC,
    state: np.ndarray | torch.Tensor,
    action_lo: np.ndarray | torch.Tensor,
    action_hi: np.ndarray | torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Records the variance of the actions sampled by the actor and the sampler.
    """
    state = ensure_2d_tensor(state)
    action_lo = ensure_2d_tensor(action_lo)
    action_hi = ensure_2d_tensor(action_hi)
    assert state.size(0) == 1

    ar = agent.get_actor_actions(
        cfg.n_samples,
        state,
        action_lo,
        action_hi,
    )
    sampled_policy_actions = ar.policy_actions.squeeze(0)
    actor_sample_var = torch.var(sampled_policy_actions, dim=0)

    ar = agent.get_sampler_actions(
        cfg.n_samples,
        state,
        action_lo,
        action_hi,
    )
    sampled_policy_actions = ar.policy_actions.squeeze(0)
    sampler_sample_var = torch.var(sampled_policy_actions, dim=0)
    return actor_sample_var, sampler_sample_var

def policy_variance(
        app_state: AppState,
        agent: BaseAgent,
        state: np.ndarray | torch.Tensor,
        action_lo: np.ndarray | torch.Tensor,
        action_hi: np.ndarray | torch.Tensor,
        ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.policy_variance,
        eval_fn=_policy_variance,
        metric_names=['actor_var', 'sampler_var'],
        state=state,
        action_lo=action_lo,
        action_hi=action_hi,
    )

# ------------------------------ Q Values Online ----------------------------- #

@config()
class QOnlineConfig:
    enabled: bool = True

def _q_online(
    cfg: QOnlineConfig,
    agent: GreedyAC,
    state: np.ndarray | torch.Tensor,
    direct_action: np.ndarray | torch.Tensor,
):
    """
    Records the Q value of the action taken, the variance across the ensemble
    and the individual Q values for the ensemble members.
    """
    state = ensure_2d_tensor(state)
    assert state.size(0) == 1
    direct_action = ensure_2d_tensor(direct_action)
    assert direct_action.size(0) == 1
    out = agent.get_values([state], [direct_action])

    return out.reduced_value, out.ensemble_values, out.ensemble_variance

def q_online(
        app_state: AppState,
        agent: BaseAgent,
        state: np.ndarray | torch.Tensor,
        direct_action: np.ndarray | torch.Tensor,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.q_online,
        eval_fn=_q_online,
        metric_names=['q', 'q_ensemble', 'q_ensemble_variance'],
        state=state,
        direct_action=direct_action,
    )

# ------------------------------ Greed Dist ------------------------------ #

def get_max_action(actions: torch.Tensor, values: torch.Tensor):
    assert actions.dim() == 3
    assert values.dim() == 2
    max_indices = torch.argmax(values, dim=1)
    batch_size = actions.size(0)
    return actions[torch.arange(batch_size), max_indices, :]

@config()
class GreedDistConfig:
    enabled: bool = True
    n_samples: int = 100

def _greed_dist(
    cfg: GreedDistConfig,
    agent: GreedyAC,
    states: np.ndarray | torch.Tensor,
    action_lo: np.ndarray | torch.Tensor,
    action_hi: np.ndarray | torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Evaluates whether the policy is greedy w.r.t. the critic in terms of a distance metric.

    For some state in the policy buffer, we sample n_samples uniformly at random.
    The greedification for this state is the l2 distance between the action that
    maximizes the Q value and the action that maximizes the policy.

    Returns the metric for each state in the batch.
    """
    states = ensure_2d_tensor(states)
    action_lo = ensure_2d_tensor(action_lo)
    action_hi = ensure_2d_tensor(action_hi)
    BATCH_SIZE = states.shape[0]
    STATE_DIM = states.shape[1]
    ACTION_DIM = agent.action_dim
    N_SAMPLES = cfg.n_samples

    def uniform_sampler(
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ):
        # Generate uniform actions in the range [action_lo, action_hi]
        actions = torch.rand(
            (BATCH_SIZE, n_samples, ACTION_DIM),
            device=device.device,
        )
        return actions * (action_hi - action_lo) + action_lo

    qr = get_sampled_qs(
        states=states,
        action_lo=action_lo,
        action_hi=action_hi,
        n_samples=N_SAMPLES,
        sampler=uniform_sampler,
        critic=agent,
    )

    q_values = qr.q_values
    max_actions_critic = get_max_action(qr.actions, q_values).reshape(BATCH_SIZE, ACTION_DIM)

    # Get log probabilities for the sampled actions from the actor.
    sampled_actions_2d = qr.actions.reshape(BATCH_SIZE * N_SAMPLES, ACTION_DIM)
    repeated_states_2d = qr.states.reshape(BATCH_SIZE * N_SAMPLES, STATE_DIM)
    log_prob_2d = agent.prob(
        repeated_states_2d,
        sampled_actions_2d,
    )

    # Get the max direct action according to log_probs for each state
    max_actions_actor = get_max_action(qr.actions, log_prob_2d).reshape(BATCH_SIZE, ACTION_DIM)

    # Calculate the L2 distance between the actions that maximize q values and the actions that maximize policy
    diff_actions = max_actions_critic - max_actions_actor
    distances = torch.norm(diff_actions, dim=1)

    return (distances,)

def greed_dist_online(
        app_state: AppState,
        agent: BaseAgent,
        states: np.ndarray | torch.Tensor,
        action_lo: np.ndarray | torch.Tensor,
        action_hi: np.ndarray | torch.Tensor,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_dist_online,
        eval_fn=_greed_dist,
        metric_names=['greed_dist_online'],
        states=states,
        action_lo=action_lo,
        action_hi=action_hi,
    )


def greed_dist_batch(app_state: AppState, agent: BaseAgent):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_dist_batch,
        eval_fn=policy_buffer_batchify(_greed_dist),
        metric_names=['greed_dist_batch'],
    )

# ------------------------------- Greed Values ------------------------------ #

@config()
class GreedValuesConfig:
    enabled: bool = True
    n_samples: int = 200
    percentile : float  | None = .1

def _greed_values(
    cfg : GreedValuesConfig,
    agent: GreedyAC,
    states: np.ndarray | torch.Tensor,
    action_lo: np.ndarray | torch.Tensor,
    action_hi: np.ndarray | torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Returns the average q value of actions sampled from the policy that are above a threshold
    determined by the top percentile of q values where actions are sampled according to the sampler.
    """
    n_samples = cfg.n_samples
    percentile = cfg.percentile
    if percentile is None:
        percentile = agent.actor_percentile

    states = ensure_2d_tensor(states)
    action_lo = ensure_2d_tensor(action_lo)
    action_hi = ensure_2d_tensor(action_hi)

    qr_sampler = get_sampled_qs(
        states, action_lo, action_hi, n_samples,
        sampler=agent.get_sampler_actions,
        critic=agent,
    )
    q_values_sampler = qr_sampler.q_values
    percentile_q_threshold = get_percentile_threshold(q_values_sampler, percentile)

    qr_actor = get_sampled_qs(
        states, action_lo, action_hi, n_samples,
        sampler=agent.get_actor_actions,
        critic=agent,
    )
    q_values_actor = qr_actor.q_values

    x = torch.mean(q_values_actor - percentile_q_threshold.unsqueeze(1), dim=1)
    return (x,)

def greed_values_online(
        app_state: AppState,
        agent: BaseAgent,
        states: np.ndarray | torch.Tensor,
        action_lo: np.ndarray | torch.Tensor,
        action_hi: np.ndarray | torch.Tensor,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_percent_online,
        eval_fn=_greed_values,
        metric_names=['greed_values_online'],
        states=states,
        action_lo=action_lo,
        action_hi=action_hi,
    )

def greed_values_batch(app_state: AppState, agent: BaseAgent):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_percent_batch,
        eval_fn=policy_buffer_batchify(_greed_values),
        metric_names=['greed_values_batch'],
    )

# ------------------------------ Q and PDF Plots ----------------------------- #

@config()
class QPDFPlotsConfig:
    enabled: bool = True
    # number of samples for the primary action (i.e. the values on the x-axis)
    primary_action_samples: int = 100
    # number of samples for other actions (i.e. how many times to average to create each point on the y-axis)
    other_action_samples: int = 10

def ac_eval(
    app_state: AppState,
    agent: GreedyAC,
    state: np.ndarray | torch.Tensor,
    action_lo: np.ndarray | torch.Tensor,
    action_hi: np.ndarray | torch.Tensor,
):
    x_axis_actions = 101
    on_policy_samples = 5

    abs_state = State(
        features=jnp.asarray(state),
        a_lo=jnp.asarray(action_lo),
        a_hi=jnp.asarray(action_hi),
        dp=jnp.ones((state.shape[0], 1), dtype=jnp.bool),
        last_a=jnp.asarray(action_lo),
    )

    # Actor and Critic evaluated at evenly spaced points along the x-axis
    linspaced_actions = jnp.linspace(start=0, stop=1, num=x_axis_actions, endpoint=True)
    repeat_linspace = linspaced_actions.repeat(on_policy_samples)

    # To evaluate critic at a given point along x-axis, use average over sampled actions for remaining action dims
    repeat_state = jax.tree.map(
        lambda x: jnp.tile(x, (on_policy_samples, 1)), state,
    )

    on_policy_actions = agent.get_actions(repeat_state)
    repeat_on_policy = jnp.tile(on_policy_actions, (x_axis_actions, 1))

    # Actor
    actor_probs = agent.get_probs(agent._actor_state.actor.params, abs_state, linspaced_actions)

    for a_dim in range(agent.action_dim):
        constructed_actions = repeat_on_policy.at[:, a_dim].set(repeat_linspace)

        # Critic
        q_vals = agent.get_action_values(abs_state, constructed_actions)
        reshaped_q_vals = q_vals.reshape((x_axis_actions, on_policy_samples))
        q_vals_over_other_a = reshaped_q_vals.mean(axis=1)

        # Actor
        a_dim_actor_probs = actor_probs[:, a_dim]

        measure = XYEval(data=[
            XY(x=x, y=float(y))
            for x, y in zip(linspaced_actions, a_dim_actor_probs, strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"pdf_plot_action_{a_dim}",
            value=measure.model_dump_json(),
        )

        # Write to db
        measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(linspaced_actions, q_vals_over_other_a, strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"qs_plot_action_{a_dim}",
            value=measure.model_dump_json(),
        )


def q_values_and_act_prob(
    app_state: AppState,
    agent: GreedyAC,
    state: np.ndarray | torch.Tensor,
    action_lo: np.ndarray | torch.Tensor,
    action_hi: np.ndarray | torch.Tensor,
):
    """
    Logs the probability density function of the policy and the Q values.
    This entries are of the form (metric, x, y) where
    * metric tells us the action being varied (think x-axis)
    * x is a direct action
    * y is the probability or Q value, averaged over samples where the action in metric is set to x
        and the other actions are sampled from the policy. cfg.other_action_samples controls the number of samples
    """
    cfg = app_state.cfg.eval_cfgs.q_pdf_plots
    if not cfg.enabled:
        return

    state = ensure_2d_tensor(state)
    action_lo = ensure_2d_tensor(action_lo)
    action_hi = ensure_2d_tensor(action_hi)

    # sample actions for the actor
    n_samples = cfg.primary_action_samples * cfg.other_action_samples
    dist = agent.get_dist(state)
    rng = jax.random.PRNGKey(app_state.agent_step)
    actions: jax.Array = dist.sample(seed=rng, sample_shape=n_samples)
    actions = actions.transpose(1, 0, 2)

    # get actions for each action dimension we are interested in.
    lin_spaced_actions = np.linspace(0, 1, cfg.primary_action_samples)
    # since we are averaging across samples for the other action dimensions, repeat these samples
    repeated_lin_spaced_actions = np.tile(lin_spaced_actions, cfg.other_action_samples)

    repeated_states = state.repeat(n_samples, 1)
    for a_dim_idx in range(agent.action_dim):
        # augmented actions are the actions we are interested in,
        # but with the primary action dimension replaced with the lin_spaced actions
        augmented_policy_actions = np.asarray(actions, copy=True)
        augmented_policy_actions[:, :, a_dim_idx] = repeated_lin_spaced_actions
        probs = dist.prob(jnp.asarray(augmented_policy_actions.squeeze(0)))
        probs = probs.reshape(
            cfg.primary_action_samples,
            cfg.other_action_samples,
        ).mean(axis=1)
        direct_actions = augmented_policy_actions

        lin_spaced_actions_in_da_space = np.unique(direct_actions[:, :, a_dim_idx], axis=0)[0]
        measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(lin_spaced_actions_in_da_space, probs, strict=False)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"pdf_plot_action_{a_dim_idx}",
            value=measure.model_dump_json(),
        )

        # Next, plot q values for the entire range of direct actions
        augmented_direct_actions = direct_actions.copy()
        augmented_direct_actions[:, :, a_dim_idx] = repeated_lin_spaced_actions
        qs = agent.get_values(
            [repeated_states],
            [torch.asarray(augmented_direct_actions.reshape(n_samples, -1))],
        ).reduced_value
        qs = qs.reshape(
            cfg.primary_action_samples,
            cfg.other_action_samples,
        ).mean(dim=1)

        measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(lin_spaced_actions, qs, strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"qs_plot_action_{a_dim_idx}",
            value=measure.model_dump_json(),
        )


class XY(BaseModel):
    x: float
    y: float

class XYEval(BaseModel):
    data: list[XY]
