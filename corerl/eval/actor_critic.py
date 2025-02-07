import json
import logging
import random
from copy import deepcopy
from typing import Tuple, cast

import torch
from torch import Tensor
from torch.distributions.constraints import interval

from corerl.agent.base import BaseAC, BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.data_pipeline.transition_filter import call_filter
from corerl.state import AppState

logger = logging.getLogger(__name__)

@config()
class ActorCriticEvalConfig:
    name: str = "actor-critic"
    enabled: bool = False
    num_test_states: int = 30
    num_uniform_actions: int = 100
    critic_samples: int = 5

class ActorCriticEval:
    def __init__(
        self,
        cfg: ActorCriticEvalConfig,
        app_state: AppState,
        agent: BaseAgent,
        column_desc: ColumnDescriptions
    ):
        self.cfg = cfg
        self.enabled = cfg.enabled
        if not isinstance(agent, BaseAC) and self.enabled:
            self.enabled = False
            logger.error("Agent must be a BaseAC to use Actor-Critic evaluator")

        self.num_test_states = cfg.num_test_states
        self.num_uniform_actions = cfg.num_uniform_actions
        self.critic_samples = cfg.critic_samples
        agent = cast(BaseAC, agent)
        self.agent = agent
        self.app_state = app_state
        self.col_desc = column_desc
        self.test_states: list[Tensor] | None = None

    def _get_a_dim_range(self) -> Tensor:
        """
        Produce evenly spaced action values along a given action dimension's support.
        The policy's probability density and the critic will be evaluated
        at each of these evenly spaced action values
        """
        support = self.agent.actor.policy.support
        assert isinstance(support, interval)
        support_low = float(support.lower_bound)
        support_high = float(support.upper_bound)
        a_dim_spacing = (support_high - support_low) / self.num_uniform_actions
        a_dim_range = torch.arange(support_low, support_high, a_dim_spacing)

        return a_dim_range

    def _get_repeat_state_actions(
        self,
        state: Tensor,
        sampled_actions: Tensor,
        a_dim: int,
        a_dim_range: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        For each a_dim value in a_dim_range, construct an action using the other action dims in sampled_actions.
        Each of these constructed actions will be evaluated at 'state'
        """
        # Construct Actions
        repeat_a_dim_range = a_dim_range.repeat((self.critic_samples, 1)).transpose(0, 1).flatten().reshape((-1, 1))
        sampled_actions_copy = deepcopy(sampled_actions)
        repeated_sample_actions = sampled_actions_copy.repeat((len(a_dim_range), 1))
        constructed_actions = torch.cat(
            (repeated_sample_actions[:, :a_dim], repeat_a_dim_range, repeated_sample_actions[:, a_dim + 1:]), 1)

        repeat_state = state.repeat((len(constructed_actions), 1))

        return repeat_state, constructed_actions

    def _get_pdf(self, repeat_state: Tensor, constructed_actions: Tensor) -> list[list[float]]:
        """
        Estimate the policy's probability density function (pdf) over a given action dimension at the given state.
        The constructed actions have values sampled from the policy for the other action dimensions.
        """
        log_probs, _ = self.agent.actor.policy.log_prob(repeat_state, constructed_actions)
        pdf_vals = torch.exp(log_probs)
        pdf_vals = pdf_vals.reshape((-1, self.critic_samples)).transpose(0, 1)

        return pdf_vals.detach().tolist()

    def _get_action_vals(self, repeat_state: Tensor, constructed_actions: Tensor) -> list[list[float]]:
        """
        Estimate the Q-function at the given state over a given action dimension.
        The constructed actions have values sampled from the policy for the other action dimensions.
        """
        qs = self.agent.q_critic.get_q([repeat_state], [constructed_actions])
        same_action_qs = qs.reshape((-1, self.critic_samples)).transpose(0, 1)

        return same_action_qs.tolist()

    def get_test_states(self, transitions: list[Transition]):
        """
        Getting test states from offline transitions.
        Will only evaluate these test states when execute_offline() is called.
        """
        # Only want to evaluate policy at states that are decision points
        dp_transitions = call_filter(transitions, 'only_post_dp')
        assert len(dp_transitions) > 0

        self.test_states = [
            transition.post.state for transition in
            random.sample(dp_transitions, self.num_test_states)
        ]

    def execute_offline(self, iter_num: int):
        if self.test_states is None:
            logger.error("Call ActorCriticEval.get_test_states() before calling ActorCriticEval.execute_offline()."
                         "len(self.test_states) must be greater than 0")
            return

        self.execute(self.test_states, str(iter_num))

    def execute(self, states: list[Tensor], label: str = ""):
        if not self.enabled:
            return

        qs_and_policy = {}
        qs_and_policy["state_cols"] = self.col_desc.state_cols
        qs_and_policy["states"] = {}
        for state in states:
            state_key = json.dumps(state.tolist())
            qs_and_policy["states"][state_key] = {}

            # When evaluating the critic over a given action dim, need sampled actions for the other action dims
            repeat_state = state.repeat((self.critic_samples, 1))
            sampled_actions, _ = self.agent.actor.get_action(repeat_state, with_grad=False)

            # Determine the interval of values that the current action dimension will be evaluated at
            a_dim_range = self._get_a_dim_range()

            # Get policy's pdf and evaluate critic over each action dim
            for a_dim in range(self.agent.action_dim):
                action_tag = self.col_desc.action_cols[a_dim]
                qs_and_policy["states"][state_key][action_tag] = {}
                qs_and_policy["states"][state_key][action_tag]["actions"] = a_dim_range.tolist()

                state_copies, built_actions = self._get_repeat_state_actions(state, sampled_actions, a_dim, a_dim_range)

                # Get policy pdf
                pdfs = self._get_pdf(state_copies, built_actions)
                qs_and_policy["states"][state_key][action_tag]["pdf"] = pdfs

                # Evaluate critic at each value of 'a_dim' in 'a_dim_range'
                same_action_qs = self._get_action_vals(state_copies, built_actions)
                qs_and_policy["states"][state_key][action_tag]["critic"] = same_action_qs

        self.app_state.evals.write(self.app_state.agent_step, f"actor-critic_{label}", qs_and_policy)
