from collections.abc import Sequence
from functools import partial
from typing import Callable, Concatenate, ParamSpec, Protocol, TypeVar

import numpy as np
import torch

from corerl.agent.base import BaseAC, BaseAgent
from corerl.agent.greedy_ac import (
    GreedyAC,
    get_percentile_threshold,
    get_q_for_sample,
    sample_actions,
    unsqueeze_repeat,
)
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

    if kwargs is None:
        kwargs = {}

    metrics_tuple = eval_fn(cfg, agent, *args, **kwargs)

    if len(metrics_tuple) == 0:
        return

    assert len(metrics_tuple) == len(metric_names)

    for metric_i, metrics  in enumerate(metrics_tuple):
        metric_name = metric_names[metric_i]
        if metrics.numel() == 1:
            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=metric_name,
                value=metrics.max(),
            )
        else:
            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"{metric_name}_min",
                value=metrics.min(),
            )

            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"{metric_name}_max",
                value=metrics.max(),
            )

            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"{metric_name}_mean",
                value=metrics.mean(),
            )

            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"{metric_name}_var",
                value=metrics.var(),
            )

SAEvalFn = Callable[[T, GreedyAC, torch.Tensor, torch.Tensor], Sequence[torch.Tensor]]
BatchSAEvalFn = Callable[[EvalConfig, GreedyAC], Sequence[torch.Tensor]]
def policy_buffer_batchify(eval_fn: SAEvalFn) ->  BatchSAEvalFn:
    def batchified(cfg: EvalConfig, agent: GreedyAC):
        batches = agent.policy_buffer.sample()

        if len(batches) == 0:
            return tuple()

        assert len(batches) == 1
        batch = batches[0]
        state_batch = batch.prior.state
        direct_action_batch = agent._filter_only_direct_actions(batch.post.action)

        return eval_fn(cfg, agent, state_batch, direct_action_batch)

    return batchified


def ensure_2d_tensor(array : np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        array = torch.tensor(array, device=device.device)
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
) -> Sequence[torch.Tensor]:
    """
    Records the variance of the actions sampled by the actor and the sampler.
    """
    state = ensure_2d_tensor(state)
    assert state.size(0) == 1
    repeated_state = state.repeat_interleave(cfg.n_samples, dim=0)

    sampled_actions, _ = agent.actor.get_action(
        repeated_state,
        with_grad=False,
    )
    actor_sample_var = torch.var(sampled_actions, dim=0)

    sampled_actions, _ = agent.sampler.get_action(
        repeated_state,
        with_grad=False,
    )
    sampler_sample_var = torch.var(sampled_actions, dim=0)
    return actor_sample_var, sampler_sample_var

def policy_variance(app_state: AppState, agent: BaseAgent, state: torch.Tensor):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.policy_variance,
        eval_fn=_policy_variance,
        metric_names=['actor_var', 'sampler_var'],
        state=state,
    )
        metric_names=['actor_var', 'sampler_var'],
        state=state,
    )

# ------------------------------ Q Values Online ----------------------------- #

@config()
class QOnlineConfig:
    enabled: bool = True

def _q_online(
    cfg: QOnlineConfig,
    agent: BaseAC,
    state: np.ndarray | torch.Tensor,
    direct_action: np.ndarray | torch.Tensor,
)-> Sequence[torch.Tensor]:
    """
    Records the Q value of the action taken, the variance across the ensemble
    and the individual Q values for the ensemble members.
    """
    state = ensure_2d_tensor(state)
    assert state.size(0) == 1
    direct_action = ensure_2d_tensor(direct_action)
    assert direct_action.size(0) == 1
    q, qs = agent.q_critic.get_qs([state], [direct_action], with_grad=False)

    return q, qs

def q_online(app_state: AppState, agent: BaseAgent, state: torch.Tensor, direct_action: torch.Tensor):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.q_online,
        eval_fn=_q_online,
        metric_names=['q', 'q_ensemble'],
        state=state,
        direct_action=direct_action,
    )
        metric_names=['q', 'q_ensemble'],
        state=state,
        direct_action=direct_action,
    )
        metric_names=['q', 'q_ensemble'],
        state=state,
        direct_action=direct_action,
    )

# ------------------------------ Greed Dist ------------------------------ #

def get_max_action(actions: torch.Tensor, values: torch.Tensor):
    max_indices = torch.argmax(values, dim=1)
    max_actions = actions[torch.arange(actions.size(0)), max_indices]
    return max_actions

@config()
class GreedDistConfig:
    enabled: bool = True
    n_samples: int = 100

def _greed_dist(
    cfg: GreedDistConfig,
    agent: GreedyAC,
    states: np.ndarray | torch.Tensor,
    direct_actions: np.ndarray | torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Evaluates whether the policy is greedy w.r.t. the critic in terms of a distance metric.

    For some state in the policy buffer, we sample n_samples uniformly at random.
    The greedification for this state is the l2 distance between the action that
    maximizes the Q value and the action that maximizes the policy.

    Returns the metric for each state in the batch.
    """
    states = ensure_2d_tensor(states)
    direct_actions = ensure_2d_tensor(direct_actions)
    BATCH_SIZE = states.shape[0]
    STATE_DIM = states.shape[1]
    ACTION_DIM = agent.action_dim
    N_SAMPLES = cfg.n_samples

    # Sample N_SAMPLES actions uniformly at random
    sampled_actions, repeated_states = sample_actions(
        states,
        N_SAMPLES,
        ACTION_DIM,
        uniform_weight=1.0,
    )

    # Ensure the sampled actions are direct actions
    direct_actions = unsqueeze_repeat(direct_actions, 1, N_SAMPLES)
    sampled_direct_actions = agent._ensure_direct_action(direct_actions, sampled_actions)
    q_values_2d = get_q_for_sample(agent, repeated_states, sampled_direct_actions)

    # Get the max direct action according to q for each state
    max_actions_critic = get_max_action(sampled_direct_actions, q_values_2d)

    # Get log probabilities for the sampled actions from the actor.
    # NOTE that the log probabilities may be defined w.r.t. delta actions.
    sampled_actions_2d = sampled_actions.reshape(BATCH_SIZE * N_SAMPLES, ACTION_DIM)
    repeated_states_2d = repeated_states.reshape(BATCH_SIZE * N_SAMPLES, STATE_DIM)
    log_prob_1d, _ = agent.actor.get_log_prob(
        repeated_states_2d,
        sampled_actions_2d,
        with_grad=False,
    )
    log_prob_2d = log_prob_1d.reshape(BATCH_SIZE, N_SAMPLES)

    # Get the max direct action according to log_probs for each state
    max_actions_actor = get_max_action(sampled_direct_actions, log_prob_2d)

    # Calculate the L2 distance between the actions that maximize q values and the actions that maximize policy
    diff_actions = max_actions_critic - max_actions_actor
    distances = torch.norm(diff_actions, dim=1)

    return (distances,)

greed_dist_online = partial(
    agent_eval,
    cfg_lens = lambda app_state: app_state.cfg.eval_cfgs.greed_dist_online,
    eval_fn = _greed_dist,
    metric_names = ['greed_dist_online']
)

greed_dist_batch = partial(
    agent_eval,
    cfg_lens = lambda app_state: app_state.cfg.eval_cfgs.greed_dist_batch,
    eval_fn = policy_buffer_batchify(_greed_dist),
    metric_names = ['greed_dist_batch']
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
    states: torch.Tensor,
    direct_actions: torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Returns the average q value of actions sampled from the policy that are above a threshold
    determined by the top percentile of q values where actions are sampled according to the sampler.
    """
    n_samples = cfg.n_samples
    percentile = cfg.percentile
    if percentile is None:
        percentile = agent.rho

    states = ensure_2d_tensor(states)
    direct_actions = ensure_2d_tensor(direct_actions)

    prop_sampled_actions, prop_repeated_states = sample_actions(
        states,
        n_samples,
        agent.action_dim,
        uniform_weight=agent.uniform_sampling_percentage,
        policy=agent.sampler
    )

    direct_actions = unsqueeze_repeat(direct_actions, 1, n_samples)
    prop_sampled_direct_actions = agent._ensure_direct_action(direct_actions, prop_sampled_actions)
    prop_q_vals_2d = get_q_for_sample(agent, prop_repeated_states, prop_sampled_direct_actions)
    percentile_q_threshold = get_percentile_threshold(prop_q_vals_2d, percentile)

    act_sampled_actions, act_repeated_states = sample_actions(
        states,
        n_samples,
        agent.action_dim,
        uniform_weight=0.0,
        policy=agent.actor,
    )

    act_sampled_direct_actions = agent._ensure_direct_action(direct_actions, act_sampled_actions)
    act_q_vals_2d = get_q_for_sample(agent, act_repeated_states, act_sampled_direct_actions)

    x = torch.mean(act_q_vals_2d - percentile_q_threshold.unsqueeze(1), dim=1)
    return (x,)

greed_values_online = partial(
    agent_eval,
    cfg_lens = lambda app_state: app_state.cfg.eval_cfgs.greed_percent_online,
    eval_fn = _greed_values,
    metric_names = ['greed_values_online']
)

greed_values_batch = partial(
    agent_eval,
    cfg_lens = lambda app_state: app_state.cfg.eval_cfgs.greed_percent_batch,
    eval_fn = policy_buffer_batchify(_greed_values),
    metric_names = ['greed_values_batch']
)


