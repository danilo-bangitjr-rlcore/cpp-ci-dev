from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from collections import deque
from corerl.alerts.base import BaseAlert
import corerl.component.network.utils as utils
from corerl.utils.device import device
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import ensemble_mse
from corerl.data.data import TransitionBatch, Transition


class ActionValueAlert(BaseAlert):
    def __init__(self, cfg: DictConfig, cumulant_start_ind: int, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        if 'state_dim' not in kwargs:
            raise KeyError("Missing required argument: 'state_dim'")
        if 'action_dim' not in kwargs:
            raise KeyError("Missing required argument: 'action_dim'")

        super().__init__(cfg, cumulant_start_ind, **kwargs)
        self.cumulant_end_ind = self.cumulant_start_ind + self.get_dim()
        self.cumulant_inds = list(range(self.cumulant_start_ind, self.cumulant_end_ind))

        # Even though we're not using the agent's critic for the alert, we still need the agent's actor to update the alert's critic
        self.agent = kwargs['agent']

        # Don't use agent's Q-function because this alert's Q-function can use a different discount factor
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.q_critic = init_q_critic(cfg.critic, self.state_dim, self.action_dim)
        self.buffer = init_buffer(cfg.buffer)

        self.gamma = cfg.gamma
        self.ret_perc = cfg.ret_perc  # Percentage of the full return being neglected in the observed partial return
        self.return_steps = int(np.ceil(np.log(self.ret_perc) / np.log(self.gamma)))
        self.trace_decay = cfg.trace_decay
        self.trace_thresh = cfg.trace_thresh
        self.ensemble_targets = cfg.ensemble_targets

        self.partial_returns = deque([], self.return_steps)
        self.values = deque([], self.return_steps)
        self.alert_trace = 0.0

    def update_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def compute_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.n_step_cumulants
        reward_batch = reward_batch[:, self.cumulant_inds]
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        gamma_exp_batch = batch.gamma_exponent
        dp_mask = batch.boot_state_dp

        next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
        with torch.no_grad():
            next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

        if self.ensemble_targets:
            _, next_q = self.q_critic.get_qs_target(
                next_state_batch, next_actions,
            )
        else:
            next_q = self.q_critic.get_q_target(next_state_batch, next_actions)

        target = reward_batch + mask_batch * (self.gamma ** gamma_exp_batch) * next_q
        _, q_ens = self.q_critic.get_qs(state_batch, action_batch, with_grad=True)
        return ensemble_mse(target, q_ens)

    def update(self) -> None:
        batch = self.buffer.sample()
        q_loss = self.compute_critic_loss(batch)
        self.q_critic.update(q_loss)

    def evaluate(self, **kwargs) -> dict:
        if 'state' not in kwargs:
            raise KeyError("Missing required argument: 'state'")
        if 'action' not in kwargs:
            raise KeyError("Missing required argument: 'action'")
        if 'reward' not in kwargs:
            raise KeyError("Missing required argument: 'reward'")

        state = kwargs['state']
        action = kwargs['action']
        reward = kwargs['reward']

        state = np.expand_dims(state, 0)
        action = np.expand_dims(action, 0)

        # Get action-value estimate for the given state-action pair
        state = utils.tensor(state, device)
        action = utils.tensor(action, device)
        value = self.q_critic.get_q(state, action, with_grad=False)
        value = utils.to_np(value)
        # The critic predicts the expected full return. Need to take into account that we're neglecting some percentage of the return.
        corrected_value = value * (1.0 - self.ret_perc)
        self.values.appendleft(corrected_value)

        # Update observed partial returns
        self.partial_returns.appendleft(0.0)
        np_partial_returns = np.array(self.partial_returns)
        num_entries = len(self.partial_returns)
        curr_reward = reward * np.ones(num_entries)
        gammas = np.array([self.gamma ** i for i in range(num_entries)])
        np_partial_returns += curr_reward * gammas
        self.partial_returns = deque(np_partial_returns, self.return_steps)

        # Only check for an alert if at least self.return_steps have elapsed
        info = self.initialize_alert_info()
        if len(self.partial_returns) == self.return_steps:
            abs_diff = abs(self.partial_returns[-1] - self.values[-1])
            # If the Observed Partial Return is greater than Q(s,a), the agent is performing the task even better than expected so set the absolute difference to 0
            if self.partial_returns[-1] > self.values[-1]:
                abs_diff = np.array(0.0)
            self.alert_trace = ((1.0 - self.trace_decay) * abs_diff) + (self.trace_decay * self.alert_trace)

            for cumulant_name in self.get_cumulant_names():
                info["alert_trace"][self.alert_type()][cumulant_name].append(self.alert_trace.squeeze().astype(float).item())
                info["alert"][self.alert_type()][cumulant_name].append(bool((self.alert_trace > self.trace_thresh).squeeze()))
                info["value"][self.alert_type()][cumulant_name].append(self.values[-1].squeeze().astype(float).item())
                info["return"][self.alert_type()][cumulant_name].append(
                    self.partial_returns[-1].squeeze().astype(float).item())

        return info

    def get_dim(self) -> int:
        return 1

    def get_discount_factors(self) -> list[float]:
        return [self.gamma]

    def alert_type(self) -> str:
        return "action_value"

    def get_cumulant_names(self) -> list[str]:
        return ["Reward"]

    def get_cumulants(self, **kwargs) -> list[float]:
        if 'reward' not in kwargs:
            raise KeyError("Missing required argument: 'reward'")

        return [kwargs['reward']]

    def get_trace_thresh(self) -> dict:
        trace_threshes = {}
        trace_threshes[self.alert_type()] = {}
        for cumulant_name in self.get_cumulant_names():
            trace_threshes[self.alert_type()][cumulant_name] = self.trace_thresh

        return trace_threshes

    def initialize_alert_info(self) -> dict:
        info = {}

        info["value"] = {}
        info["return"] = {}
        info["alert_trace"] = {}
        info["alert"] = {}

        for key in info:
            info[key][self.alert_type()] = {}
            for cumulant_name in self.get_cumulant_names():
                info[key][self.alert_type()][cumulant_name] = []

        return info
