import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor

from corerl.agent.greedy_ac import GreedyAC
from corerl.component.policy_manager import ActionReturn
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.pipeline import ColumnDescriptions, Pipeline
from corerl.data_pipeline.transition_filter import call_filter
from corerl.state import AppState

logger = logging.getLogger(__name__)

@dataclass
class Bounds:
    """
    Upper and lower bounds for each action dimension in either the direct or delta action case
    """
    low: np.ndarray
    high: np.ndarray

class PDF(BaseModel):
    """
    Probability Density Function
    """
    x_range: list[float]
    pdfs: list[list[float]]
    loc: float
    scale: float

class QFunc(BaseModel):
    x_range: list[float]
    q_vals: list[list[float]]

class ActionPlotInfo(BaseModel):
    action_tag: str
    action_val: float
    pdf: PDF
    delta_critic: QFunc | None
    direct_critic: QFunc

class StatePlotInfo(BaseModel):
    state: list[float]
    current_action: list[float]
    a_dims: list[ActionPlotInfo]

class PlotInfoBatch(BaseModel):
    state_cols: list[str]
    action_cols: list[str]
    states: list[StatePlotInfo]

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
        agent: GreedyAC,
        column_desc: ColumnDescriptions
    ):
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.delta_action = app_state.cfg.feature_flags.delta_actions

        self.num_test_states = cfg.num_test_states
        self.num_uniform_actions = cfg.num_uniform_actions
        self.critic_samples = cfg.critic_samples
        self.agent = agent
        self.app_state = app_state
        self.pipeline = pipeline
        self.col_desc = column_desc
        self.test_states: list[Tensor] | None = None
        self.test_actions: list[Tensor] | None = None

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

    def _get_policy_state_actions(
        self,
        state: Tensor,
        ar: ActionReturn,
        a_dim: int,
        a_dim_range: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        For each a_dim value in a_dim_range, construct an action using the other action dims in the sampled actions.
        The sampled actions and the values in a_dim_range are all in the space of [0,1] values output by the actor
        and are not necessarily the values in the normalized direct action space.
        The probability densities will be determined for each of these constructed actions at the given state
        """
        # Construct Actions
        repeat_a_dim_range = a_dim_range.repeat((self.critic_samples, 1)).transpose(0, 1).flatten().reshape((-1, 1))
        sampled_actions_copy = deepcopy(ar.policy_actions)
        repeated_sample_actions = sampled_actions_copy.repeat((len(a_dim_range), 1))
        constructed_actions = torch.cat(
            (repeated_sample_actions[:, :a_dim], repeat_a_dim_range, repeated_sample_actions[:, a_dim + 1:]), 1)

        repeat_state = state.repeat((len(constructed_actions), 1))

        return repeat_state, constructed_actions

    def _get_critic_state_actions(
        self,
        state: Tensor,
        ar: ActionReturn,
        a_dim: int,
        a_dim_range: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        For each a_dim value in a_dim_range, construct an action using the other action dims in the sampled actions.
        The sampled actions and the values in a_dim_range are all in the normalized direct action space.
        The Q-values will be determined for each of these constructed actions at the given state
        """
        # Construct Actions
        repeat_a_dim_range = a_dim_range.repeat((self.critic_samples, 1)).transpose(0, 1).flatten().reshape((-1, 1))
        sampled_actions_copy = deepcopy(ar.direct_actions)
        repeated_sample_actions = sampled_actions_copy.repeat((len(a_dim_range), 1))
        constructed_actions = torch.cat(
            (repeated_sample_actions[:, :a_dim], repeat_a_dim_range, repeated_sample_actions[:, a_dim + 1:]), 1)

        repeat_state = state.repeat((len(constructed_actions), 1))

        return repeat_state, constructed_actions

    def _get_pdf(self, repeat_state: Tensor, constructed_actions: Tensor) -> Tuple[list[list[float]], dict]:
        """
        Estimate the policy's probability density function (pdf) over a given action dimension at the given state.
        The constructed actions have values sampled from the policy for the other action dimensions.
        """
        log_probs, param_info = self.agent.log_prob(repeat_state, constructed_actions)
        pdf_vals = torch.exp(log_probs)
        pdf_vals = pdf_vals.reshape((-1, self.critic_samples)).transpose(0, 1)

        return pdf_vals.detach().tolist(), param_info

    def _get_action_vals(self, repeat_state: Tensor, constructed_actions: Tensor) -> list[list[float]]:
        """
        Estimate the Q-function at the given state over a given action dimension.
        The constructed actions have values sampled from the policy for the other action dimensions.
        """
        qs = self.agent.critic.get_values([repeat_state], [constructed_actions])
        same_action_qs = qs.reduced_value.reshape((-1, self.critic_samples)).transpose(0, 1)

        return same_action_qs.tolist()

    def _get_delta_action_bounds(
        self,
        curr_action_np: np.ndarray
    ) -> Bounds:
        """
        Get the min and max possible direct actions
        for each action dimension given the current action
        """
        delta_low = self.agent._policy_manager.delta_low.numpy()
        delta_high = self.agent._policy_manager.delta_high.numpy()

        return Bounds(np.clip(curr_action_np + delta_low, 0, 1), np.clip(curr_action_np + delta_high, 0, 1))

    def _get_policy_plot_info(
        self,
        state: Tensor,
        ar: ActionReturn,
        a_dim: int,
        policy_range: Tensor,
        x_range: list[float]
    ) -> PDF:
        state_copies, built_actions = self._get_policy_state_actions(state,
                                                                     ar,
                                                                     a_dim,
                                                                     policy_range)
        densities, param_info = self._get_pdf(state_copies, built_actions)
        loc_np = param_info['loc'][0]
        loc = float(loc_np[a_dim])
        scale_np = param_info['scale'][0]
        scale = float(scale_np[a_dim])
        pdf = PDF(x_range=x_range,
                  pdfs=densities,
                  loc=loc,
                  scale=scale)

        return pdf

    def _get_critic_plot_info(
        self,
        state: Tensor,
        ar: ActionReturn,
        a_dim: int,
        x_range: Tensor
    ) -> QFunc:
        state_copies, built_actions = self._get_critic_state_actions(state,
                                                                     ar,
                                                                     a_dim,
                                                                     x_range)
        same_action_qs = self._get_action_vals(state_copies, built_actions)
        q_func = QFunc(x_range=x_range.tolist(),
                       q_vals=same_action_qs)

        return q_func

    def _get_delta_action_plot_info(
        self,
        state: Tensor,
        curr_action: Tensor,
        ar: ActionReturn
    ) -> StatePlotInfo:
        action_dim = len(self.col_desc.action_cols)
        policy_bounds = Bounds(low=np.zeros(action_dim), high=np.ones(action_dim))
        direct_bounds = Bounds(low=np.zeros(action_dim), high=np.ones(action_dim))
        curr_action_np = curr_action.numpy()
        delta_bounds = self._get_delta_action_bounds(curr_action_np)

        action_plot_info_l = []
        for a_dim in range(action_dim):
            action_tag = self.col_desc.action_cols[a_dim]

            # Get policy pdf and critic q-values over delta action range
            policy_range = torch.linspace(policy_bounds.low[a_dim],
                                          policy_bounds.high[a_dim],
                                          self.num_uniform_actions)
            delta_x_range = torch.linspace(delta_bounds.low[a_dim],
                                           delta_bounds.high[a_dim],
                                           self.num_uniform_actions)
            pdf = self._get_policy_plot_info(state,
                                             ar,
                                             a_dim,
                                             policy_range,
                                             delta_x_range.tolist())
            delta_q = self._get_critic_plot_info(state,
                                                 ar,
                                                 a_dim,
                                                 delta_x_range)

            # Get critic q-values over full direct action range
            direct_x_range = torch.linspace(direct_bounds.low[a_dim],
                                            direct_bounds.high[a_dim],
                                            self.num_uniform_actions)
            direct_q = self._get_critic_plot_info(state,
                                                  ar,
                                                  a_dim,
                                                  direct_x_range)

            action_plot_info = ActionPlotInfo(action_tag=action_tag,
                                              action_val=float(curr_action[a_dim]),
                                              pdf=pdf,
                                              delta_critic=delta_q,
                                              direct_critic=direct_q)
            action_plot_info_l.append(action_plot_info)


        state_plot_info = StatePlotInfo(state=state.tolist(),
                                        current_action=curr_action.tolist(),
                                        a_dims=action_plot_info_l)

        return state_plot_info

    def _get_direct_action_plot_info(
        self,
        state: Tensor,
        curr_action: Tensor,
        ar: ActionReturn
    ) -> StatePlotInfo:
        action_plot_info_l = []
        action_dim = len(self.col_desc.action_cols)
        bounds = Bounds(low=np.zeros(action_dim), high=np.ones(action_dim))
        for a_dim in range(action_dim):
            action_tag = self.col_desc.action_cols[a_dim]

            # Get critic q-values and policy pdf over full direct action range
            a_dim_range = torch.linspace(bounds.low[a_dim],
                                         bounds.high[a_dim],
                                         self.num_uniform_actions)
            pdf = self._get_policy_plot_info(state,
                                             ar,
                                             a_dim,
                                             a_dim_range,
                                             a_dim_range.tolist())
            direct_q = self._get_critic_plot_info(state,
                                                  ar,
                                                  a_dim,
                                                  a_dim_range)
            action_plot_info = ActionPlotInfo(action_tag=action_tag,
                                              action_val=float(curr_action[a_dim]),
                                              pdf=pdf,
                                              delta_critic=None,
                                              direct_critic=direct_q)
            action_plot_info_l.append(action_plot_info)

        state_plot_info = StatePlotInfo(state=state.tolist(),
                                        current_action=curr_action.tolist(),
                                        a_dims=action_plot_info_l)

        return state_plot_info

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

        state_plot_info_l = []
        for i in range(len(states)):
            state = states[i]
            prev_action = actions[i]

            # When evaluating the critic over a given action dim, need sampled actions for the other action dims
            repeat_state = state.repeat((self.critic_samples, 1))
            repeat_prev_action = prev_action.repeat((self.critic_samples, 1))
            ar = self.agent.get_actor_actions(repeat_state, repeat_prev_action)

            # Produce different plots for direct action and delta action agents
            if self.delta_action:
                state_plot_info = self._get_delta_action_plot_info(state, prev_action, ar)
            else:
                state_plot_info = self._get_direct_action_plot_info(state, prev_action, ar)

            state_plot_info_l.append(state_plot_info)

        plot_info_batch = PlotInfoBatch(state_cols=self.col_desc.state_cols,
                                        action_cols=self.col_desc.action_cols,
                                        states=state_plot_info_l)
        plot_info_dump = plot_info_batch.model_dump_json()
        self.app_state.evals.write(self.app_state.agent_step, f"actor-critic_{label}", plot_info_dump)
