import logging
import random
from collections import namedtuple
from copy import deepcopy
from typing import Tuple, cast

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor

from corerl.agent.base import BaseAC, BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.pipeline import ColumnDescriptions, Pipeline
from corerl.data_pipeline.transforms.delta import Delta
from corerl.data_pipeline.transition_filter import call_filter
from corerl.state import AppState

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
        if not isinstance(agent, BaseAC) and self.enabled:
            self.enabled = False
            logger.error("Agent must be a BaseAC to use Actor-Critic evaluator")

        self.num_test_states = cfg.num_test_states
        self.num_uniform_actions = cfg.num_uniform_actions
        self.critic_samples = cfg.critic_samples
        agent = cast(BaseAC, agent)
        self.agent = agent
        self.app_state = app_state
        self.pipeline = pipeline
        self.col_desc = column_desc
        self.action_cols = [col for col in self.col_desc.action_cols if not Delta.is_delta_transformed(col)]
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
        log_probs, _ = self.agent.actor.policy.log_prob(repeat_state, constructed_actions)
        pdf_vals = torch.exp(log_probs)
        pdf_vals = pdf_vals.reshape((-1, self.critic_samples)).transpose(0, 1)

        return pdf_vals.detach().tolist()

    def _get_action_vals(self, repeat_state: Tensor, constructed_actions: Tensor) -> list[list[float]]:
        """
        Estimate the Q-function at the given state over a given action dimension.
        The constructed actions have values sampled from the policy for the other action dimensions.
        """
        qs = self.agent.critic.get_q([repeat_state], [constructed_actions])
        same_action_qs = qs.reshape((-1, self.critic_samples)).transpose(0, 1)

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

    def _get_norm_action(self, action_arr: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Get the normalized direct action from an offset and delta
        """
        norm_delta_df = self.pipeline.action_constructor.assign_action_names(action_arr, delta)
        norm_action = norm_delta_df.to_numpy()[0]

        return norm_action

    def _get_delta_action_bounds(
        self,
        curr_action_np: np.ndarray
    ) -> Bounds:
        """
        Get the min and max possible delta actions for each action dimension given the current action offset
        """
        low_delta = np.zeros(len(self.action_cols))
        norm_delta_low = self._get_norm_action(curr_action_np, low_delta)
        high_delta = np.ones(len(self.action_cols))
        norm_delta_high = self._get_norm_action(curr_action_np, high_delta)

        return Bounds(norm_delta_low, norm_delta_high)

    def _get_direct_action_bounds(
        self,
    ) -> Bounds:
        """
        Get the min and max possible direct actions for each action dimension
        """
        if self.app_state.cfg.agent.delta_action:
            dummy = np.ones(len(self.action_cols)) * 0.5
            low_direct = np.zeros(len(self.col_desc.action_cols))
            norm_direct_low = self._get_norm_action(low_direct, dummy)
            high_direct = np.ones(len(self.col_desc.action_cols))
            norm_direct_high = self._get_norm_action(high_direct, dummy)
        else:
            dummy = np.zeros(len(self.action_cols))
            low_direct = np.zeros(len(self.col_desc.action_cols))
            norm_direct_low = self._get_norm_action(dummy, low_direct)
            high_direct = np.ones(len(self.col_desc.action_cols))
            norm_direct_high = self._get_norm_action(dummy, high_direct)

        return Bounds(norm_direct_low, norm_direct_high)

    def _get_policy_plot_info(
        self,
        qs_and_policy: dict,
        state: Tensor,
        sampled_actions: Tensor,
        a_dim: int,
        norm_a_dim_range: Tensor,
    ) -> dict:
        state_key = State(state=state.tolist()).model_dump_json()
        action_tag = self.action_cols[a_dim]
        state_copies, built_actions = self._get_repeat_state_actions(state,
                                                                     sampled_actions,
                                                                     a_dim,
                                                                     norm_a_dim_range)

        pdfs = self._get_pdf(state_copies, built_actions)
        qs_and_policy["plot_info"][state_key]["a_dim"][action_tag]["pdf"] = pdfs
        qs_and_policy["plot_info"][state_key]["a_dim"][action_tag]["policy_actions"] = norm_a_dim_range.tolist()

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
        action_tag = self.action_cols[a_dim]
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
        sampled_actions: Tensor,
        qs_and_policy: dict,
    ) -> dict:
        state_key = State(state=state.tolist()).model_dump_json()
        curr_action_np = curr_action.numpy()

        norm_delta_low, norm_delta_high = self._get_delta_action_bounds(curr_action_np)
        norm_direct_low, norm_direct_high = self._get_direct_action_bounds()

        for a_dim in range(len(self.action_cols)):
            action_tag = self.action_cols[a_dim]
            qs_and_policy["plot_info"][state_key]["a_dim"][action_tag] = {}

            # Get policy pdf and critic q-values over delta action range
            norm_delta_a_dim_range = torch.linspace(float(norm_delta_low[a_dim]),
                                                    float(norm_delta_high[a_dim]),
                                                    self.num_uniform_actions)
            qs_and_policy = self._get_policy_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_actions,
                                                       a_dim,
                                                       norm_delta_a_dim_range)
            qs_and_policy = self._get_critic_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_actions,
                                                       a_dim,
                                                       norm_delta_a_dim_range,
                                                       "delta")

            # Get critic q-values over full direct action range
            norm_direct_a_dim_range = torch.linspace(float(norm_direct_low[a_dim]),
                                                     float(norm_direct_high[a_dim]),
                                                     self.num_uniform_actions)
            qs_and_policy = self._get_critic_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_actions,
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

        norm_direct_low, norm_direct_high = self._get_direct_action_bounds()

        for a_dim in range(self.agent.action_dim):
            action_tag = self.action_cols[a_dim]
            qs_and_policy["plot_info"][state_key]["a_dim"][action_tag] = {}

            # Get critic q-values and policy pdf over full direct action range
            norm_direct_a_dim_range = torch.linspace(float(norm_direct_low[a_dim]),
                                                     float(norm_direct_high[a_dim]),
                                                     self.num_uniform_actions)
            qs_and_policy = self._get_policy_plot_info(qs_and_policy,
                                                       state,
                                                       sampled_actions,
                                                       a_dim,
                                                       norm_direct_a_dim_range)
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
        qs_and_policy["action_cols"] = self.action_cols
        qs_and_policy["plot_info"] = {}
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            state_key = State(state=state.tolist()).model_dump_json()
            action_dump = Action(action=action.tolist()).model_dump_json()
            qs_and_policy["plot_info"][state_key] = {}
            qs_and_policy["plot_info"][state_key]["current_action"] = action_dump
            qs_and_policy["plot_info"][state_key]["a_dim"] = {}

            # When evaluating the critic over a given action dim, need sampled actions for the other action dims
            repeat_state = state.repeat((self.critic_samples, 1))
            sampled_actions, _ = self.agent.actor.get_action(repeat_state, with_grad=False)

            # Produce different plots for direct action and delta action agents
            if self.app_state.cfg.agent.delta_action:
                qs_and_policy = self._get_delta_action_plot_info(state, action, sampled_actions, qs_and_policy)
            else:
                qs_and_policy = self._get_direct_action_plot_info(state, sampled_actions, qs_and_policy)

        self.app_state.evals.write(self.app_state.agent_step, f"actor-critic_{label}", qs_and_policy)
