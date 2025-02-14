import numpy as np
import torch

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import GreedyAC
from corerl.component.network.utils import state_to_tensor, tensor
from corerl.configs.config import config
from corerl.state import AppState
from corerl.utils.device import device


@config()
class PolicyVarianceConfig:
    enabled: bool = True
    n_samples: int = 100


def eval_policy_variance(
        app_state: AppState,
        state: np.ndarray,
        agent : BaseAgent,
    ):
    """
    Records the variance of the actions sampled by the actor and the sampler.
    Only supported for GAC currently.
    """
    cfg = app_state.cfg.eval_cfgs.policy_variance

    if not cfg.enabled:
        return

    if not isinstance(agent, GreedyAC):
        return

    tensor_state = state_to_tensor(state, device.device)
    repeated_state = tensor_state.repeat_interleave(cfg.n_samples, dim=0)

    # log var for the actor
    sampled_actions, _ = agent.actor.get_action(
        repeated_state,
        with_grad=False,
    )
    sample_var = torch.var(sampled_actions, dim=0)
    mean_var = torch.mean(sample_var)
    app_state.metrics.write(
        agent_step=app_state.agent_step,
        metric="actor_action_var",
        value=mean_var,
    )

    # log var for the sampler
    sampled_actions, _ = agent.sampler.get_action(
        repeated_state,
        with_grad=False,
    )
    sample_var = torch.var(sampled_actions, dim=0)
    mean_var = torch.mean(sample_var)
    app_state.metrics.write(
        agent_step=app_state.agent_step,
        metric="sampler_action_var",
        value=mean_var,
    )


@config()
class QOnlineConfig:
    enabled: bool = True

def eval_q_online(
        app_state: AppState,
        state: np.ndarray,
        agent : BaseAgent,
        action : np.ndarray,
    ):
    """
    Records the Q value of the action taken, the variance across the ensemble
    and the infividual Q values for the ensemble members.
    Only supported for GAC currently.
    """
    cfg = app_state.cfg.eval_cfgs.q_online

    if not cfg.enabled:
        return

    if not isinstance(agent, GreedyAC):
        return

    action_tensor = tensor(action, device.device)
    tensor_state = state_to_tensor(state, device.device)

    q, qs = agent.q_critic.get_qs([tensor_state], [action_tensor], with_grad=False)

    app_state.metrics.write(
        agent_step=app_state.agent_step,
        metric="q",
        value=q,
    )

    app_state.metrics.write(
        agent_step=app_state.agent_step,
        metric="q_var",
        value=torch.var(qs) if len(qs) > 1 else 0,
    )

    for i, q_i in enumerate(qs):
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric=f"q_{i}",
            value=q_i,
        )

