import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from lib_agent.buffer.buffer import State
from lib_config.config import MISSING, computed, config

from corerl.agent.greedy_ac import GreedyAC
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
        return cfg.agent.gamma


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

    def __init__(self, cfg: MonteCarloEvalConfig, app_state: AppState, agent: GreedyAC):
        self.cfg = cfg
        self.enabled = cfg.enabled

        # Determine partial return horizon
        self.gamma = cfg.gamma
        assert 0 <= self.gamma < 1.0
        self.precision = cfg.precision
        if self.gamma == 0:
            self.return_steps = 1
        else:
            self.return_steps = math.ceil(np.log(1.0 - self.precision) / np.log(self.gamma))

        # Queue to compute partial returns and temporally align partial returns with corresponding Q-values
        self._step_queue = deque[_MonteCarloPoint](maxlen=self.return_steps)
        self.critic_samples = cfg.critic_samples

        self.prev_state: State | None = None  # to deal with one step offset between states and actions
        self.agent_step = 0
        self.app_state = app_state
        self.agent = agent

    def _get_state_value(self, state: State):
        """
        Estimates the given state's value under the agent's current policy
        by evaluating the agent's Q function at the given state
        under a few actions sampled from the agent's policy and averaging them.
        Returns a given state's value when the partial return horizon has elapsed.
        """
        sampled_actions = self.agent.get_actions(state, n=self.critic_samples)

        # Get reduced Q-values and average them across sampled actions
        sampled_a_qs = self.agent.get_values(
            state.features,
            sampled_actions,
        ).reduced_value
        return float(sampled_a_qs.mean())


    def _get_partial_return(self):
        """
        Iteratively computes the partial returns of sequential states over a horizon of self.return_steps
        one reward at a time.
        Returns a computed partial return once the horizon of self.return_steps has elapsed.
        """
        if len(self._step_queue) < self.return_steps:
            return None

        partial_return = 0.0
        gamma = self.gamma ** (self.return_steps - 1)
        for step in self._step_queue:
            partial_return += gamma * step.reward
            gamma /= self.gamma

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

    def execute(
            self,
            curr_state: State,
            curr_time: datetime,
            reward: float,
            label: str = ''):

        if not self.enabled:
            return

        if self.prev_state is None:
            self.prev_state = curr_state
            return

        # Can't compute partial returns or evaluate critic if there are nans in the state, action, or reward
        if (jnp.isnan(self.prev_state.features).any()
            or jnp.isnan(curr_state.last_a).any()):
            self._step_queue.clear()
            self.prev_state = curr_state
            return

        state_v = self._get_state_value(self.prev_state)
        observed_a_q = (self.agent.get_values(self.prev_state.features, curr_state.last_a)
                        .reduced_value.item())

        self._step_queue.appendleft(_MonteCarloPoint(
            timestamp=curr_time.isoformat(),
            agent_step=self.app_state.agent_step,
            state_v=state_v,
            observed_a_q=observed_a_q,
            reward=reward,
        ))

        partial_return = self._get_partial_return()

        if partial_return is not None:
            step = self._step_queue.pop()
            self._write_metrics(step, partial_return, label)

        self.prev_state = curr_state
