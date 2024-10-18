from omegaconf import DictConfig

import numpy as np
import torch
from collections import deque
from corerl.alerts.base import BaseAlert
import corerl.component.network.utils as utils
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np
from corerl.data.data import TransitionBatch, Transition


class ActionValueTraceAlert(BaseAlert):
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

        # Even though we're not using the agent's critic for the alert,
        # we still need the agent's actor to update the alert's critic
        self.agent = kwargs['agent']

        # Don't use agent's Q-function because this alert's Q-function can use a different discount factor
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.q_critic = init_q_critic(cfg.critic, self.state_dim, self.action_dim)
        self.buffer = init_buffer(cfg.buffer)

        self.gamma = cfg.gamma
        self.ret_perc = cfg.ret_perc  # Percentage of the full return being neglected in the observed partial return
        self.return_steps = int(np.ceil(np.log(self.ret_perc) / np.log(self.gamma)))
        self.alert_trace_decay = cfg.alert_trace_decay
        self.alert_trace_thresh = cfg.alert_trace_thresh
        self.ensemble_targets = cfg.ensemble_targets

        self.partial_returns = deque([], self.return_steps)
        self.values = deque([], self.return_steps)
        self.alert_trace = 0.0

    def update_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def load_buffer(self, transitions: list[Transition]) -> None:
        self.buffer.load(transitions)

    def compute_critic_loss(self, ensemble_batch: list[TransitionBatch]) -> tuple[list[torch.Tensor], dict]:
        ensemble = len(ensemble_batch)
        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []
        next_action_batches = []
        mask_batches = []
        gamma_exp_batches = []
        next_qs = []
        for batch in ensemble_batch:
            state_batch = batch.state
            action_batch = batch.action
            reward_batch = batch.n_step_cumulants
            reward_batch = reward_batch[:, self.cumulant_inds]
            next_state_batch = batch.boot_state
            mask_batch = 1 - batch.terminated
            gamma_exp_batch = batch.gamma_exponent
            dp_mask = batch.boot_state_dp

            next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
            # For the 'Anytime' paradigm, only states at decision points can sample next_actions
            # If a state isn't at a decision point, its next_action is set to the current action
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

            # Option 1: Using the reduction of the ensemble in the update target
            if not self.ensemble_targets:
                next_q = self.q_critic.get_q_target([next_state_batch], [next_actions])
                next_qs.append(next_q)

            state_batches.append(state_batch)
            action_batches.append(action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            next_action_batches.append(next_actions)
            mask_batches.append(mask_batch)
            gamma_exp_batches.append(gamma_exp_batch)

        # Option 2: Using the corresponding target function in the ensemble in the update target
        if self.ensemble_targets:
            _, next_qs = self.q_critic.get_qs_target(next_state_batches, next_action_batches)
        else:
            for i in range(ensemble):
                next_qs[i] = torch.unsqueeze(next_qs[i], 0)
            next_qs = torch.cat(next_qs, dim=0)

        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=True)
        losses = []
        for i in range(ensemble):
            # N-Step SARSA update with variable 'N', thus 'reward_batch' is an n_step reward
            # and the exponent on gamma, 'gamma_exp_batch', depends on 'n'
            target = reward_batches[i] + mask_batches[i] * (self.gamma ** gamma_exp_batches[i]) * next_qs[i]
            losses.append(torch.nn.functional.mse_loss(target, qs[i]))

        ensemble_info = {}
        return losses, ensemble_info

    def update(self) -> dict:
        batches = self.buffer.sample()

        def closure():
            losses, ensemble_info = self.compute_critic_loss(batches)
            loss = torch.stack(losses, dim=-1).sum(dim=-1)
            return loss, ensemble_info

        q_loss, ens_info = closure()
        self.q_critic.update(q_loss, opt_kwargs={"closure": closure})

        return ens_info

    def update_partial_returns(self, reward: float) -> None:
        self.partial_returns.appendleft(0.0)
        np_partial_returns = np.array(self.partial_returns)
        num_entries = len(self.partial_returns)
        curr_reward = reward * np.ones(num_entries)
        gammas = np.array([self.gamma ** i for i in range(num_entries)])
        np_partial_returns += curr_reward * gammas
        self.partial_returns = deque(np_partial_returns, self.return_steps)

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
        state = utils.tensor(state)
        action = utils.tensor(action)
        value = self.q_critic.get_q([state], [action], with_grad=False)
        value = utils.to_np(value)
        # The critic predicts the expected full return.
        # Need to take into account that we're neglecting some percentage of the return.
        corrected_value = value * (1.0 - self.ret_perc)
        self.values.appendleft(corrected_value)

        # Update observed partial returns
        self.update_partial_returns(reward)

        # Only check for an alert if at least self.return_steps have elapsed
        info = self.initialize_alert_info()
        if len(self.partial_returns) == self.return_steps:
            abs_diff = np.abs(self.partial_returns[-1] - self.values[-1]).item()
            # If the Observed Partial Return is greater than Q(s,a),
            # the agent is performing the task even better than expected so set the absolute difference to 0
            if self.partial_returns[-1] > self.values[-1]:
                abs_diff = 0.0
            self.alert_trace = ((1.0 - self.alert_trace_decay) * abs_diff) + (self.alert_trace_decay * self.alert_trace)
            for cumulant_name in self.get_cumulant_names():
                info["alert_trace"][self.alert_type()][cumulant_name].append(self.alert_trace)
                info["alert"][self.alert_type()][cumulant_name].append(bool(self.alert_trace > self.alert_trace_thresh))
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
            trace_threshes[self.alert_type()][cumulant_name] = self.alert_trace_thresh

        return trace_threshes

    def get_buffer_size(self):
        return self.buffer.size

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

    def get_test_state_qs(self, plot_info, repeated_test_states, repeated_actions, num_states, test_actions):
        # Q-Values
        q_values, ensemble_qs = self.q_critic.get_qs([repeated_test_states], [repeated_actions], with_grad=False)
        q_values = to_np(q_values)
        q_values = q_values.reshape(num_states, test_actions, self.get_dim())
        ensemble_qs = to_np(ensemble_qs)
        ensemble_qs = ensemble_qs.reshape(self.q_critic.model.ensemble, num_states, test_actions, self.get_dim())

        cumulant_names = self.get_cumulant_names()
        for i in range(len(cumulant_names)):
            cumulant_name = cumulant_names[i]
            plot_info["q_values"][cumulant_name] = q_values[:,:,i]
            plot_info["ensemble_qs"][cumulant_name] = ensemble_qs[:,:,:,i]

        return plot_info

class ActionValueUncertaintyAlert(ActionValueTraceAlert):
    def __init__(self, cfg: DictConfig, cumulant_start_ind: int, **kwargs):
        super().__init__(cfg, cumulant_start_ind, **kwargs)

        self.std_trace_decay = cfg.std_trace_decay
        self.std_trace_thresh = cfg.std_trace_thresh
        self.std_trace = 0.0

        self.stds = deque([], self.return_steps)
        self.means = deque([], self.return_steps)

        self.active_alert = False

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

        # Get ensemble action-value estimates for the given state-action pair
        state = utils.tensor(state)
        action = utils.tensor(action)
        q, q_ens = self.q_critic.get_qs([state], [action], with_grad=False)
        q_ens = utils.to_np(q_ens)
        q_std = q_ens.std()
        self.stds.appendleft(q_std)
        q_mean = q_ens.mean()
        self.means.appendleft(q_mean)
        # The critic predicts the expected full return.
        # Need to take into account that we're neglecting some percentage of the return.
        q = utils.to_np(q)
        corrected_q = q * (1.0 - self.ret_perc)
        self.values.appendleft(corrected_q)

        # Update observed partial returns
        self.update_partial_returns(reward)

        # Only check for an alert if at least self.return_steps have elapsed
        info = self.initialize_alert_info()
        if len(self.partial_returns) == self.return_steps:
            curr_return = self.partial_returns[-1].squeeze().astype(float).item()
            curr_std = self.stds[-1].squeeze().astype(float).item()
            curr_value = self.values[-1].squeeze().astype(float).item()

            # Always update the STD trace
            self.std_trace = ((1.0 - self.std_trace_decay) * curr_std) + (self.std_trace_decay * self.std_trace)

            # Only update the alert trace when the STD trace is below its threshold
            if self.std_trace < self.std_trace_thresh:
                abs_diff = abs(curr_return - curr_value)
                # If the Observed Partial Return is greater than Q(s,a),
                # the agent is performing the task even better than expected so set the absolute difference to 0
                if curr_return > curr_value:
                    abs_diff = 0.0

                decay = self.alert_trace_decay
                self.alert_trace = ((1.0 - decay) * abs_diff) + (decay * self.alert_trace)

            self.active_alert = bool(self.alert_trace > self.alert_trace_thresh)

            for cumulant_name in self.get_cumulant_names():
                # info["std"][self.alert_type()][cumulant_name].append(curr_std / np.absolute(curr_mean))
                info["std_trace"][self.alert_type()][cumulant_name].append(self.std_trace)
                info["alert_trace"][self.alert_type()][cumulant_name].append(self.alert_trace)
                info["alert"][self.alert_type()][cumulant_name].append(self.active_alert)
                info["return"][self.alert_type()][cumulant_name].append(curr_return)
                info["value"][self.alert_type()][cumulant_name].append(curr_value)

        return info

    def initialize_alert_info(self) -> dict:
        info = {}

        info["value"] = {}
        info["return"] = {}
        info["alert_trace"] = {}
        info["alert"] = {}
        info["std_trace"] = {}

        for key in info:
            info[key][self.alert_type()] = {}
            for cumulant_name in self.get_cumulant_names():
                info[key][self.alert_type()][cumulant_name] = []

        return info

    def get_std_thresh(self) -> dict:
        std_threshes = {}
        std_threshes[self.alert_type()] = {}
        for cumulant_name in self.get_cumulant_names():
            std_threshes[self.alert_type()][cumulant_name] = self.std_trace_thresh

        return std_threshes
