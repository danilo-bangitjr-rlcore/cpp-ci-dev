import logging
import math
from collections import deque
from typing import Tuple, cast

import numpy as np
from torch import Tensor

from corerl.agent.base import BaseAC, BaseAgent
from corerl.component.network.utils import tensor
from corerl.configs.config import config, interpolate, list_
from corerl.data_pipeline.pipeline import PipelineReturn
from corerl.state import AppState

logger = logging.getLogger(__name__)


@config()
class MonteCarloEvalConfig:
    enabled: bool = False
    offline_eval_steps: list[int] = list_()
    gamma: float = interpolate('${experiment.gamma}')
    precision: float = 0.99 # Monte-Carlo return within 'precision'% of the true return (can't compute infinite sum)
    critic_samples: int = 5

class MonteCarloEvaluator:
    """
    Iteratively computes the observed partial returns for the states in the given PipelineFrame.
    Estimates an observed state's Q-value under the agent's policy over actions sampled from the agent's policy
    as well as under the observed action to compare against the observed partial returns.
    """

    def __init__(self, cfg: MonteCarloEvalConfig, app_state: AppState, agent: BaseAgent, pipe_return: PipelineReturn):
        self.cfg = cfg
        self.enabled = cfg.enabled

        if not isinstance(agent, BaseAC) and self.enabled:
            self.enabled = False
            logger.error("Agent must be a BaseAC to use Monte-Carlo evaluator")

        # Determine partial return horizon
        self.gamma = cfg.gamma
        self.precision = cfg.precision
        self.return_steps = math.ceil(np.log(1.0 - self.precision) / np.log(self.gamma))

        # Queues to compute partial returns and temporally align partial returns with corresponding Q-values
        self.timestamps = deque(maxlen=self.return_steps)
        self.agent_steps = deque(maxlen=self.return_steps)
        self.mc_partial_returns = deque(maxlen=self.return_steps)
        self.observed_a_qs = deque(maxlen=self.return_steps)
        self.state_vs = deque(maxlen=self.return_steps)
        self.critic_samples = cfg.critic_samples

        self.agent_step = 0
        self.app_state = app_state
        self.agent = cast(BaseAC, agent)
        assert len(pipe_return.states) > 0, \
            "Monte-Carlo Evaluator must have states, actions, and rewards dataframes with one or more entries"
        self.pipe_return: PipelineReturn = pipe_return

    def _reset_queues(self):
        self.timestamps = deque(maxlen=self.return_steps)
        self.mc_partial_returns = deque(maxlen=self.return_steps)
        self.observed_a_qs = deque(maxlen=self.return_steps)
        self.state_vs = deque(maxlen=self.return_steps)

    def _get_timestamp(self, timestamp: str) -> Tuple[str | None, int | None]:
        self.timestamps.appendleft(timestamp)
        self.agent_steps.appendleft(self.agent_step)

        if len(self.timestamps) == self.return_steps:
            return self.timestamps.pop(), self.agent_steps.pop()
        else:
            return None, None

    def _get_state_value(self, state: Tensor) -> float | None:
        """
        Estimates the given state's value under the agent's current policy
        by evaluating the agent's Q function at the given state
        under a few actions sampled from the agent's policy and averaging them.
        Returns a given state's value when the partial return horizon has elapsed.
        """
        repeat_state = state.repeat((self.critic_samples, 1))
        sampled_actions, _ = self.agent.actor.get_action(repeat_state, with_grad=False)
        sampled_a_qs = self.agent.q_critic.get_q([repeat_state],
                                                 [sampled_actions],
                                                 with_grad=False,
                                                 bootstrap_reduct=True)
        sampled_a_avg_q = float(sampled_a_qs.mean())
        self.state_vs.appendleft(sampled_a_avg_q)

        if len(self.state_vs) == self.return_steps:
            return self.state_vs.pop()
        else:
            return None

    def _get_observed_a_q(self, state: Tensor, observed_a: Tensor) -> float | None:
        """
        Returns the agent's action-value estimate for the given state-action pair
        under the agent's current policy.
        Returns a given state's value when the partial return horizon has elapsed.
        """
        observed_a_q = self.agent.q_critic.get_q([state.expand(1, -1)],
                                                 [observed_a.expand(1, -1)],
                                                 with_grad=False,
                                                 bootstrap_reduct=True)
        self.observed_a_qs.appendleft(float(observed_a_q.flatten()))

        if len(self.observed_a_qs) == self.return_steps:
            return self.observed_a_qs.pop()
        else:
            return None

    def _get_partial_return(self, reward: float) -> float | None:
        """
        Iteratively computes the partial returns of sequential states over a horizon of self.return_steps
        one reward at a time.
        Returns a computed partial return once the horizon of self.return_steps has elapsed.
        """
        self.mc_partial_returns.appendleft(0.0)
        discounts = (np.ones(len(self.mc_partial_returns)) * self.gamma) ** np.arange(len(self.mc_partial_returns))
        deltas = discounts * (np.ones(len(self.mc_partial_returns)) * reward)
        self.mc_partial_returns += deltas
        self.mc_partial_returns = deque(self.mc_partial_returns, maxlen=self.return_steps)

        if len(self.mc_partial_returns) == self.return_steps:
            return self.mc_partial_returns.pop()
        else:
            return None

    def _write_metrics(self,
                       train_iter: int,
                       timestamp: str,
                       agent_step: int,
                       state_v: float,
                       observed_a_q: float,
                       partial_return: float):
        self.app_state.metrics.write(metric=f"state_v_{train_iter}",
                                     value=state_v,
                                     timestamp=timestamp,
                                     agent_step=agent_step)
        self.app_state.metrics.write(metric=f"observed_a_q_{train_iter}",
                                     value=observed_a_q,
                                     timestamp=timestamp,
                                     agent_step=agent_step)
        self.app_state.metrics.write(metric=f"partial_return_{train_iter}",
                                     value=partial_return,
                                     timestamp=timestamp,
                                     agent_step=agent_step)

    def __call__(self, iter_num: int):
        if not self.enabled or iter_num not in self.cfg.offline_eval_steps:
            return

        states = self.pipe_return.states
        taken_actions = self.pipe_return.actions
        rewards = self.pipe_return.rewards.to_numpy().astype(np.float32).flatten()
        # To get the action taken and the reward observed from the given state,
        # need to offset actions and rewards by one obs_period with respect to the state
        taken_actions = taken_actions[1:]
        rewards = rewards[1:]
        for i in range(len(rewards)):
            state = tensor(states.iloc[i].to_numpy())
            observed_a = tensor(taken_actions.iloc[i].to_numpy())
            reward: float = float(rewards[i])
            # Can't compute partial returns or evaluate critic if there are nans in the state, action, or reward
            if any([np.isnan(t).any() for t in [state, observed_a, reward]]):
                self._reset_queues()
                self.agent_step += 1
                continue

            # Get Timestamp
            curr_time = states.index[i].isoformat()
            timestamp, agent_step = self._get_timestamp(curr_time)

            # Get Policy Action Q-Value
            state_v = self._get_state_value(state)

            # Get Observed Action Q-Value
            observed_a_q = self._get_observed_a_q(state, observed_a)

            # Get Partial Return
            partial_return = self._get_partial_return(reward)

            if all(v is not None for v in [timestamp, state_v, observed_a_q, partial_return]):
                assert timestamp is not None
                assert agent_step is not None
                assert state_v is not None
                assert observed_a_q is not None
                assert partial_return is not None
                self._write_metrics(iter_num, timestamp, agent_step, state_v, observed_a_q, partial_return)

            self.agent_step += 1
