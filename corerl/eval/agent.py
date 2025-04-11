from collections.abc import Sequence
from typing import Callable, Concatenate, ParamSpec, Protocol, SupportsFloat, TypeVar

import numpy as np
import torch
from pydantic import BaseModel

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
        sampler=agent.get_actor_actions,
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

# ------------------------------ Q and PDF Plots ----------------------------- #

@config()
class QPDFPlotsConfig:
    enabled: bool = True
    # number of samples for the primary action (i.e. the values on the x-axis)
    primary_action_samples: int = 100
    # number of samples for other actions (i.e. how many times to average to create each point on the y-axis)
    other_action_samples: int = 100

def q_values_and_act_prob(
        app_state: AppState,
        agent: BaseAgent,
        state: np.ndarray | torch.Tensor,
        prev_action: np.ndarray | torch.Tensor
        ):
    """
    Logs the probability density function of the policy and the Q values.
    This entries are of the form (metric, x, y) where
    * metric tells us the action being varied (think x-axis)
    * x is a direct action
    * y is the probability or Q value, averaged over samples where the action in metric is set to x
        and the other actions are sampled from the policy. cfg.other_action_samples controls the number of samples
    """
    if not isinstance(agent, GreedyAC):
        return
    cfg = app_state.cfg.eval_cfgs.q_pdf_plots
    if not cfg.enabled:
        return

    state = ensure_2d_tensor(state)
    prev_action = ensure_2d_tensor(prev_action)
    repeated_states = state.repeat(
        cfg.primary_action_samples*cfg.other_action_samples, 1)
    repeated_prev_a = prev_action.repeat(
        cfg.primary_action_samples*cfg.other_action_samples, 1)

    # sample actions for the actor
    ar = agent.get_actor_actions(repeated_states, repeated_prev_a)

    # get actions for each action dimension we are interested in.
    lin_spaced_actions = torch.linspace(
        0, 1, cfg.primary_action_samples, device=device.device)
    # since we are averaging across samples for the other action dimensions, repeat these samples
    repeated_lin_spaced_actions = torch.repeat_interleave(
        lin_spaced_actions, cfg.other_action_samples)

    for a_dim_idx in range(agent.action_dim):
        # augmented actions are the actions we are interested in,
        # but with the primary action dimension replaced with the lin_spaced actions
        augmented_policy_actions = ar.policy_actions.clone()
        augmented_policy_actions[:, a_dim_idx] = repeated_lin_spaced_actions
        # convert these policy actions to direct action
        direct_actions = agent.policy_to_direct_action(repeated_prev_a, augmented_policy_actions)
        direct_actions = torch.clip(direct_actions, 0, 1)
        probs = agent.prob(
            repeated_states,
            augmented_policy_actions, # probability is calculated using policy actions
        )
        probs = probs.reshape(
            cfg.primary_action_samples,
            cfg.other_action_samples,
        ).mean(dim=1)

        lin_spaced_actions_in_da_space = direct_actions[:, a_dim_idx].unique()
        measure = XYEval(data=[
            XY(x=x, y=float(y))
            for x, y in zip(lin_spaced_actions_in_da_space, probs, strict=False)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"pdf_plot_action_{a_dim_idx}",
            value=measure.model_dump_json(),
        )

        # Next, plot q values for the entire range of direct actions
        augmented_direct_actions = direct_actions.clone()
        augmented_direct_actions[:, a_dim_idx] = repeated_lin_spaced_actions
        qs = agent.critic.get_values(
            [repeated_states],
            [augmented_direct_actions],
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
