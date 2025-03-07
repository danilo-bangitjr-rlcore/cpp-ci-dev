from collections.abc import Sequence
from typing import Callable, Concatenate, ParamSpec, Protocol, SupportsFloat, TypeVar

import numpy as np
import torch

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import (
    GreedyAC,
)
from corerl.agent.utils import get_percentile_threshold, get_sampled_qs
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

SAEvalFn = Callable[[T, GreedyAC, torch.Tensor, torch.Tensor], Sequence[torch.Tensor]]
BatchSAEvalFn = Callable[[EvalConfig, GreedyAC], Sequence[torch.Tensor]]
def policy_buffer_batchify(eval_fn: SAEvalFn) ->  BatchSAEvalFn:
    def batchified(cfg: EvalConfig, agent: GreedyAC):
        if not agent.is_policy_buffer_sampleable:
            return tuple()

        batches = agent.sample_policy_buffer()
        assert len(batches) == 1
        batch = batches[0]
        state_batch = batch.prior.state
        action_batch = batch.post.action

        return eval_fn(cfg, agent, state_batch, action_batch)

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
    prev_action: np.ndarray | torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Records the variance of the actions sampled by the actor and the sampler.
    """
    state = ensure_2d_tensor(state)
    prev_action = ensure_2d_tensor(prev_action)
    assert state.size(0) == 1
    assert prev_action.size(0) == 1
    repeated_state = state.repeat_interleave(cfg.n_samples, dim=0)
    repeated_prev_action = prev_action.repeat_interleave(cfg.n_samples, dim=0)

    ar = agent.get_actor_actions(
        repeated_state,
        repeated_prev_action,
    )
    sampled_policy_actions = ar.policy_actions
    actor_sample_var = torch.var(sampled_policy_actions, dim=0)

    ar = agent.get_sampler_actions(
        repeated_state,
        repeated_prev_action,
    )
    sampled_policy_actions = ar.policy_actions
    sampler_sample_var = torch.var(sampled_policy_actions, dim=0)
    return actor_sample_var, sampler_sample_var

def policy_variance(
        app_state: AppState,
        agent: BaseAgent,
        state: np.ndarray | torch.Tensor,
        prev_action: np.ndarray | torch.Tensor,
        ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.policy_variance,
        eval_fn=_policy_variance,
        metric_names=['actor_var', 'sampler_var'],
        state=state,
        prev_action=prev_action,
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
    out = agent.critic.get_values([state], [direct_action], with_grad=False)

    return out

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
        metric_names=['q', 'q_ensemble'],
        state=state,
        direct_action=direct_action,
    )

# ------------------------------ Greed Dist ------------------------------ #

def get_max_action(actions: torch.Tensor, values: torch.Tensor):
    assert actions.dim() == 3
    assert values.dim() == 2
    max_indices = torch.argmax(values, dim=1)
    batch_size = actions.size(0)
    max_actions = actions[torch.arange(batch_size), max_indices, :]
    return max_actions

@config()
class GreedDistConfig:
    enabled: bool = True
    n_samples: int = 100

def _greed_dist(
    cfg: GreedDistConfig,
    agent: GreedyAC,
    states: np.ndarray | torch.Tensor,
    prev_actions: np.ndarray | torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Evaluates whether the policy is greedy w.r.t. the critic in terms of a distance metric.

    For some state in the policy buffer, we sample n_samples uniformly at random.
    The greedification for this state is the l2 distance between the action that
    maximizes the Q value and the action that maximizes the policy.

    Returns the metric for each state in the batch.
    """
    states = ensure_2d_tensor(states)
    prev_actions = ensure_2d_tensor(prev_actions)
    BATCH_SIZE = states.shape[0]
    STATE_DIM = states.shape[1]
    ACTION_DIM = agent.action_dim
    N_SAMPLES = cfg.n_samples

    qr = get_sampled_qs(
        states, prev_actions, N_SAMPLES,
        sampler=agent.get_uniform_actions,
        critic=agent.critic,
    )

    q_values = qr.q_values
    max_actions_critic = get_max_action(qr.direct_actions, q_values).reshape(BATCH_SIZE, ACTION_DIM)

    # Get log probabilities for the sampled actions from the actor.
    sampled_policy_actions_2d = qr.policy_actions.reshape(BATCH_SIZE * N_SAMPLES, ACTION_DIM)
    repeated_states_2d = qr.states.reshape(BATCH_SIZE * N_SAMPLES, STATE_DIM)
    with torch.no_grad():
        log_prob_1d, _ = agent.log_prob(
            repeated_states_2d,
            sampled_policy_actions_2d,
        )
    log_prob_2d = log_prob_1d.reshape(BATCH_SIZE, N_SAMPLES)

    # Get the max direct action according to log_probs for each state
    max_actions_actor = get_max_action(qr.direct_actions, log_prob_2d).reshape(BATCH_SIZE, ACTION_DIM)

    # Calculate the L2 distance between the actions that maximize q values and the actions that maximize policy
    diff_actions = max_actions_critic - max_actions_actor
    distances = torch.norm(diff_actions, dim=1)

    return (distances,)

def greed_dist_online(
        app_state: AppState,
        agent: BaseAgent,
        states: np.ndarray | torch.Tensor,
        prev_actions: np.ndarray | torch.Tensor,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_dist_online,
        eval_fn=_greed_dist,
        metric_names=['greed_dist_online'],
        states=states,
        prev_actions=prev_actions,
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
    prev_actions: np.ndarray | torch.Tensor,
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
    prev_actions = ensure_2d_tensor(prev_actions)

    qr_sampler = get_sampled_qs(
        states, prev_actions, n_samples,
        sampler=agent.get_sampler_actions,
        critic=agent.critic
    )
    q_values_sampler = qr_sampler.q_values
    percentile_q_threshold = get_percentile_threshold(q_values_sampler, percentile)

    qr_actor = get_sampled_qs(
        states, prev_actions, n_samples,
        sampler=agent.get_sampler_actions,
        critic=agent.critic
    )
    q_values_actor = qr_actor.q_values

    x = torch.mean(q_values_actor - percentile_q_threshold.unsqueeze(1), dim=1)
    return (x,)

def greed_values_online(
        app_state: AppState,
        agent: BaseAgent,
        states: np.ndarray | torch.Tensor,
        prev_actions: np.ndarray | torch.Tensor
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_percent_online,
        eval_fn=_greed_values,
        metric_names=['greed_values_online'],
        states=states,
        prev_actions=prev_actions,
    )

def greed_values_batch(app_state: AppState, agent: BaseAgent):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_percent_batch,
        eval_fn=policy_buffer_batchify(_greed_values),
        metric_names=['greed_values_batch'],
    )

