import logging
import random
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Tuple, cast

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import GreedyAC
from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.pipeline import ColumnDescriptions, Pipeline
from corerl.data_pipeline.transition_filter import call_filter
from corerl.state import AppState

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)

Bounds = namedtuple('Bounds', ['low', 'high'])

class State(BaseModel):
    state: list[float]

class Action(BaseModel):
    action: list[float]

@config()
class ActorCriticEvalConfig:
    name: str = "actor-critic"
    enabled: bool = False
    num_test_states: int = 30
    num_uniform_actions: int = 100
    critic_samples: int = 5
    delta_actions : bool = MISSING

    @computed('delta_action')
    @classmethod
    def _delta_action(cls, cfg: 'MainConfig'):
        return cfg.feature_flags.delta_actions

class ActorCriticEval:
    def __init__(
        self,
        cfg: ActorCriticEvalConfig,
        app_state: AppState,
        pipeline: Pipeline,
        agent: BaseAgent,
        column_desc: ColumnDescriptions
    ):
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.delta_actions = cfg.delta_actions
        if not isinstance(agent, GreedyAC) and self.enabled:
            self.enabled = False
            logger.error("Agent must be a GreedyAC to use Actor-Critic evaluator")

        self.num_test_states = cfg.num_test_states
        self.num_uniform_actions = cfg.num_uniform_actions
        self.critic_samples = cfg.critic_samples
        agent = cast(GreedyAC, agent)
        self.agent = agent
        self.app_state = app_state
        self.pipeline = pipeline
        self.col_desc = column_desc
        self.test_states: list[Tensor] | None = None
        self.test_actions: list[Tensor] | None = None

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
        log_probs, _ = self.agent.log_prob(repeat_state, constructed_actions)
        pdf_vals = torch.exp(log_probs)
        pdf_vals = pdf_vals.reshape((-1, self.critic_samples)).transpose(0, 1)

        return pdf_vals.detach().tolist()

    def _get_action_vals(self, repeat_state: Tensor, constructed_actions: Tensor) -> list[list[float]]:
        """
        Estimate the Q-function at the given state over a given action dimension.
        The constructed actions have values sampled from the policy for the other action dimensions.
        """
        qs = self.agent.critic.get_values([repeat_state], [constructed_actions])
        same_action_qs = qs.reduced_value.reshape((-1, self.critic_samples)).transpose(0, 1)

        return same_action_qs.tolist()

    def get_test_states(self, transitions: list[Transition]):
        """
        Getting test states from offline transitions.
        Will only evaluate these test states when execute_offline() is called.
        """
        # Only want to evaluate policy at states that are decision points
        post_dp_transitions = call_filter(transitions, 'only_post_dp')
        assert len(post_dp_transitions) > 0

        test_transitions = random.sample(post_dp_transitions, self.num_test_states)

        self.test_states = [transition.post.state for transition in test_transitions]
        self.test_actions = [transition.post.action for transition in test_transitions]

    def _get_conditional_action_bounds(
        self,
        curr_action_np: np.ndarray
    ) -> Bounds:
        """
        Get the min and max possible direct actions
        for each action dimension given the current action
        """
        delta_low = self.agent._policy_manager.delta_low
        delta_high = self.agent._policy_manager.delta_high
        return Bounds(np.clip(curr_action_np+delta_low, 0, 1), np.clip(curr_action_np+delta_high, 0, 1))

    def _get_action_bounds(
        self,
    ) -> Bounds:
        """
        Get the min and max possible direct actions for each action dimension
        """
        norm_direct_low = np.zeros(len(self.col_desc.action_cols))
        norm_direct_high = np.ones(len(self.col_desc.action_cols))

        return Bounds(norm_direct_low, norm_direct_high)

    def _get_policy_plot_info(
        self,
        qs_and_policy: dict,
        state: Tensor,
        sampled_actions: Tensor,
        a_dim: int,
        policy_a_dim_range: Tensor, # the normalized a_dim_range (output by policy)
        axis_a_dim_range : Tensor # the unnormalized a_dim_range
    ) -> dict:

        state_key = State(state=state.tolist()).model_dump_json()
        action_tag = self.col_desc.action_cols[a_dim]
        state_copies, built_actions = self._get_repeat_state_actions(state,
                                                                     sampled_actions,
                                                                     a_dim,
                                                                     policy_a_dim_range)

        pdfs = self._get_pdf(state_copies, built_actions)
        qs_and_policy["plot_info"][state_key]["a_dim"][action_tag]["pdf"] = pdfs
        qs_and_policy["plot_info"][state_key]["a_dim"][action_tag]["policy_actions"] = axis_a_dim_range.tolist()

        return qs_and_policy

    def _get_critic_plot_info(
        self,
        qs_and_policy: dict,
        state: Tensor,
        sampled_actions: Tensor,
        a_dim: int,
        norm_a_dim_range: Tensor,
        label: str,
    ) -> dict:
        state_key = State(state=state.tolist()).model_dump_json()
        action_tag = self.col_desc.action_cols[a_dim]
        state_copies, built_actions = self._get_repeat_state_actions(state,
                                                                     sampled_actions,
                                                                     a_dim,
                                                                     norm_a_dim_range)

        same_action_qs = self._get_action_vals(state_copies, built_actions)
        qs_and_policy["plot_info"][state_key]["a_dim"][action_tag][f"{label}_critic"] = same_action_qs
        a_range = norm_a_dim_range.tolist()
        qs_and_policy["plot_info"][state_key]["a_dim"][action_tag][f"{label}_critic_actions"] = a_range

        return qs_and_policy

    def _get_delta_action_plot_info(
        self,
        state: Tensor,
        curr_action: Tensor,
        sampled_direct_actions: Tensor,
        sampled_policy_actions: Tensor,
        qs_and_policy: dict,
    ) -> dict:
        state_key = State(state=state.tolist()).model_dump_json()
        curr_action_np = curr_action.numpy()

        norm_delta_low, norm_delta_high = self._get_conditional_action_bounds(curr_action_np)
        norm_direct_low, norm_direct_high = self._get_action_bounds()

        for a_dim in range(len(self.col_desc.action_cols)):
            action_tag = self.col_desc.action_cols[a_dim]
            qs_and_policy["plot_info"][state_key]["a_dim"][action_tag] = {}

            # Get policy pdf and critic q-values over delta action range
            delta_a_dim_range = torch.linspace(float(norm_delta_low[a_dim]),
                                                    float(norm_delta_high[a_dim]),
                                                    self.num_uniform_actions)
            norm_delta_a_dim_range = torch.linspace(0,1,self.num_uniform_actions)
            qs_and_policy = self._get_policy_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_policy_actions,
                                                       a_dim,
                                                       norm_delta_a_dim_range,
                                                       delta_a_dim_range,
                                                       )
            qs_and_policy = self._get_critic_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_direct_actions,
                                                       a_dim,
                                                       norm_delta_a_dim_range,
                                                       "delta")

            # Get critic q-values over full direct action range
            norm_direct_a_dim_range = torch.linspace(float(norm_direct_low[a_dim]),
                                                     float(norm_direct_high[a_dim]),
                                                     self.num_uniform_actions)
            qs_and_policy = self._get_critic_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_direct_actions,
                                                       a_dim,
                                                       norm_direct_a_dim_range,
                                                       "direct")

        return qs_and_policy

    def _get_direct_action_plot_info(
        self,
        state: Tensor,
        sampled_actions: Tensor,
        qs_and_policy: dict,
    ) -> dict:
        state_key = State(state=state.tolist()).model_dump_json()

        norm_direct_low, norm_direct_high = self._get_action_bounds()

        for a_dim in range(self.agent.action_dim):
            action_tag = self.col_desc.action_cols[a_dim]
            qs_and_policy["plot_info"][state_key]["a_dim"][action_tag] = {}

            # Get critic q-values and policy pdf over full direct action range
            norm_direct_a_dim_range = torch.linspace(float(norm_direct_low[a_dim]),
                                                     float(norm_direct_high[a_dim]),
                                                     self.num_uniform_actions)
            qs_and_policy = self._get_policy_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_actions,
                                                       a_dim,
                                                       norm_direct_a_dim_range,
                                                       norm_direct_a_dim_range
                                                       )
            qs_and_policy = self._get_critic_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_actions,
                                                       a_dim,
                                                       norm_direct_a_dim_range,
                                                       "direct")

        return qs_and_policy

    def execute_offline(self, iter_num: int):
        if self.test_states is None or self.test_actions is None:
            logger.error("Call ActorCriticEval.get_test_states() before calling ActorCriticEval.execute_offline()."
                         "len(self.test_states) must be greater than 0")
            return

        self.execute(self.test_states, self.test_actions, str(iter_num))

    def execute(self, states: list[Tensor], actions: list[Tensor], label: str = ""):
        """
        Get the information needed to produce an Actor-Critic plot for either the direct action or delta action case.
        For each state S_t passed to the Actor-Critic Eval, the corresponding action A_{t-1} is also passed because
        when delta actions are being used, we need to know the offset to determine the support of the policy
        """
        if not self.enabled:
            return

        if len(states) != len(actions):
            logger.error("The Actor-Critic Eval must be passed an action for every state and vice-versa")
            return

        qs_and_policy = {}
        qs_and_policy["state_cols"] = self.col_desc.state_cols
        qs_and_policy["action_cols"] = self.col_desc.action_cols
        qs_and_policy["plot_info"] = {}
        for i in range(len(states)):
            state = states[i]
            prev_action = actions[i]
            state_key = State(state=state.tolist()).model_dump_json()
            action_dump = Action(action=prev_action.tolist()).model_dump_json()
            qs_and_policy["plot_info"][state_key] = {}
            qs_and_policy["plot_info"][state_key]["current_action"] = action_dump
            qs_and_policy["plot_info"][state_key]["a_dim"] = {}

            # When evaluating the critic over a given action dim, need sampled actions for the other action dims
            repeat_state = state.repeat((self.critic_samples, 1))
            repeat_prev_action = prev_action.repeat((self.critic_samples, 1))
            ar  = self.agent.get_actor_actions(repeat_state, repeat_prev_action)
            sampled_direct_actions = ar.direct_actions
            sampled_policy_actions = ar.policy_actions

            # Produce different plots for direct action and delta action agents
            if self.delta_actions:
                qs_and_policy = self._get_delta_action_plot_info(state, prev_action, sampled_direct_actions,
                                                                 sampled_policy_actions, qs_and_policy)
            else:
                qs_and_policy = self._get_direct_action_plot_info(state, sampled_direct_actions, qs_and_policy)

        self.app_state.evals.write(self.app_state.agent_step, f"actor-critic_{label}", qs_and_policy)
