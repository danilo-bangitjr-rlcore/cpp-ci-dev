import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from torch import Tensor

from corerl.agent.base import BaseAC, BaseAgent
from corerl.component.network.utils import tensor
from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.pipeline import PipelineReturn
from corerl.state import AppState

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)


@config()
class MonteCarloEvalConfig:
    enabled: bool = False
    precision: float = 0.99 # Monte-Carlo return within 'precision'% of the true return (can't compute infinite sum)
    critic_samples: int = 5
    gamma: float = MISSING

    @computed('gamma')
    @classmethod
    def _gamma(cls, cfg: 'MainConfig'):
        return cfg.experiment.gamma


@dataclass
class _MonteCarloPoint:
    timestamp: str
    agent_step: int
    state_v: float
    observed_a_q: float
    reward: float


class MonteCarloEvaluator:
    """
    Iteratively computes the observed partial returns for the states in the given PipelineFrame.
    Estimates an observed state's Q-value under the agent's policy over actions sampled from the agent's policy
    as well as under the observed action to compare against the observed partial returns.
    """

    def __init__(self, cfg: MonteCarloEvalConfig, app_state: AppState, agent: BaseAgent):
        self.cfg = cfg
        self.enabled = cfg.enabled

        if not isinstance(agent, BaseAC) and self.enabled:
            self.enabled = False
            logger.error("Agent must be a BaseAC to use Monte-Carlo evaluator")

        # Determine partial return horizon
        self.gamma = cfg.gamma
        self.precision = cfg.precision
        self.return_steps = math.ceil(np.log(1.0 - self.precision) / np.log(self.gamma))

        # Queue to compute partial returns and temporally align partial returns with corresponding Q-values
        self._step_queue = deque[_MonteCarloPoint](maxlen=self.return_steps)
        self.critic_samples = cfg.critic_samples

        self.agent_step = 0
        self.app_state = app_state
        self.agent = cast(BaseAC, agent)

    def _get_state_value(self, state: Tensor) -> float:
        """
        Estimates the given state's value under the agent's current policy
        by evaluating the agent's Q function at the given state
        under a few actions sampled from the agent's policy and averaging them.
        Returns a given state's value when the partial return horizon has elapsed.
        """
        repeat_state = state.repeat((self.critic_samples, 1))
        sampled_actions, _ = self.agent.actor.get_action(repeat_state, with_grad=False)
        sampled_a_qs = self.agent.critic.get_values(
            [repeat_state],
            [sampled_actions],
            with_grad=False,
        )
        sampled_a_avg_q = float(sampled_a_qs.reduced_value.mean())
        return sampled_a_avg_q

    def _get_observed_a_q(self, state: Tensor, observed_a: Tensor) -> float:
        """
        Returns the agent's action-value estimate for the given state-action pair
        under the agent's current policy.
        Returns a given state's value when the partial return horizon has elapsed.
        """
        observed_a_q = self.agent.critic.get_values(
            [state.expand(1, -1)],
            [observed_a.expand(1, -1)],
            with_grad=False,
        )
        return observed_a_q.reduced_value.item()

    def _get_partial_return(self) -> float | None:
        """
        Iteratively computes the partial returns of sequential states over a horizon of self.return_steps
        one reward at a time.
        Returns a computed partial return once the horizon of self.return_steps has elapsed.
        """
        if len(self._step_queue) < self.return_steps:
            return

        partial_return = 0.0
        gamma = 1.0
        for step in self._step_queue:
            partial_return += gamma * step.reward
            gamma *= self.gamma

        return partial_return


    def _write_metrics(
        self,
        step: _MonteCarloPoint,
        partial_return: float,
        label: str = '',
    ):
        if label:
            label = f"_{label}"
        self.app_state.metrics.write(
            metric=f"state_v{label}",
            value=step.state_v,
            timestamp=step.timestamp,
            agent_step=step.agent_step,
        )
        self.app_state.metrics.write(
            metric=f"observed_a_q{label}",
            value=step.observed_a_q,
            timestamp=step.timestamp,
            agent_step=step.agent_step,
        )
        self.app_state.metrics.write(
            metric=f"partial_return{label}",
            value=partial_return,
            timestamp=step.timestamp,
            agent_step=step.agent_step,
        )


    def execute_offline(self, iter_num: int, pipe_return: PipelineReturn):
        self.execute(pipe_return, str(iter_num))


    def execute(self, pipe_return: PipelineReturn, label: str = ''):
        if not self.enabled:
            return

        states = pipe_return.states
        taken_actions = pipe_return.actions
        rewards = pipe_return.rewards['reward'].to_numpy()
        # To get the action taken and the reward observed from the given state,
        # need to offset actions and rewards by one obs_period with respect to the state
        taken_actions = taken_actions[1:]
        rewards = rewards[1:]
        for i in range(len(rewards)):
            state = tensor(states.iloc[i].to_numpy())
            observed_a = tensor(taken_actions.iloc[i].to_numpy())
            reward = float(rewards[i])
            # Can't compute partial returns or evaluate critic if there are nans in the state, action, or reward
            if state.isnan().any() or observed_a.isnan().any() or np.isnan(reward):
                self._step_queue.clear()
                self.agent_step += 1
                continue


            curr_time = states.index[i].isoformat()
            state_v = self._get_state_value(state)
            observed_a_q = self._get_observed_a_q(state, observed_a)

            self._step_queue.appendleft(_MonteCarloPoint(
                timestamp=curr_time,
                agent_step=self.agent_step,
                state_v=state_v,
                observed_a_q=observed_a_q,
                reward=reward,
            ))

            partial_return = self._get_partial_return()

            if partial_return is not None:
                step = self._step_queue.pop()
                self._write_metrics(step, partial_return, label)

            self.agent_step += 1
