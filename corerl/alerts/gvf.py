from omegaconf import DictConfig, OmegaConf

import numpy as np
import time
from collections import deque
from corerl.alerts.base import BaseAlert
import corerl.component.network.utils as utils
from corerl.utils.device import device
from corerl.prediction.gvf import SimpleGVF, QGVF
from corerl.data.data import Transition


class GVFTraceAlert(BaseAlert):
    def __init__(self, cfg: DictConfig, cumulant_start_ind: int, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        if 'input_dim' not in kwargs:
            raise KeyError("Missing required argument: 'input_dim'")
        if 'action_dim' not in kwargs:
            raise KeyError("Missing required argument: 'action_dim'")

        super().__init__(cfg, cumulant_start_ind, **kwargs)

        self.agent = kwargs["agent"]
        self.input_dim = kwargs["input_dim"]
        self.action_dim = kwargs["action_dim"]

        self.endo_obs_names = cfg.endo_obs_names
        self.endo_inds = cfg.endo_inds
        assert len(self.endo_obs_names) > 0, "In config/env/<env_name>.yaml, define 'endo_obs_names' to be a list of the names of the endogenous variables in the observation"
        assert len(self.endo_inds) > 0, "In config/env/<env_name>.yaml, define 'endo_inds' to be a list of the indices of the endogenous variables within the environment's observation vector"
        assert len(self.endo_obs_names) == len(self.endo_inds), "The length of self.endo_obs_names and self.endo_inds should be the same and the ordering of the indices should correspond to the ordering of the variable names"

        self.gvfs = QGVF(cfg, self.input_dim, self.action_dim, agent=self.agent)
        self.num_gvfs = self.gvfs.get_num_gvfs()
        self.ensemble_targets = cfg.ensemble_targets
        self.gamma = cfg.gamma

        self.cumulant_end_ind = self.cumulant_start_ind + self.get_dim()
        self.cumulant_inds = list(range(self.cumulant_start_ind, self.cumulant_end_ind))

        self.ret_perc = cfg.ret_perc  # Percentage of the full return being neglected in the observed partial return
        self.return_steps = int(np.ceil(np.log(self.ret_perc) / np.log(self.gamma)))
        self.trace_thresh = cfg.trace_thresh
        self.trace_decay = cfg.trace_decay

        self.partial_returns = deque([], self.return_steps)
        self.values = deque([], self.return_steps)
        self.alert_trace = np.zeros(self.num_gvfs)

    def update_buffer(self, transition: Transition) -> None:
        self.gvfs.update_train_buffer(transition)

    def load_buffer(self, transitions: list[Transition]) -> None:
        self.gvfs.load_train_buffer(transitions)

    def get_buffer_size(self):
        return self.gvfs.get_buffer_size()

    def update(self) -> dict:
        update_start = time.time()
        gvf_info = self.gvfs.update(self.cumulant_inds)

        ensemble_info = {}
        ensemble_info[self.alert_type()] = {}
        ind = 0
        for cumulant_name in self.get_cumulant_names():
            ensemble_info[self.alert_type()][cumulant_name] = {}
            # Probably need to comment out the line below
            # ensemble_info[self.alert_type()][cumulant_name]["std"] = gvf_info["std"][:, ind]
            ind += 1
        update_end = time.time()
        print("GVF Alert Update Duration:", update_end - update_start)

        return ensemble_info

    def update_partial_returns(self, curr_cumulants: np.ndarray) -> None:
        self.partial_returns.appendleft([0.0 for i in range(self.num_gvfs)])
        np_partial_returns = np.array(self.partial_returns)
        num_entries = len(self.partial_returns)
        curr_cumulants = np.array([curr_cumulants for i in range(num_entries)])
        gammas = np.array([self.gamma ** i for i in range(num_entries)]).reshape(-1, 1)
        np_partial_returns += curr_cumulants * gammas
        self.partial_returns = deque(np_partial_returns, self.return_steps)

    def evaluate(self, **kwargs) -> dict:
        if 'state' not in kwargs:
            raise KeyError("Missing required argument: 'state'")
        if 'action' not in kwargs:
            raise KeyError("Missing required argument: 'action'")
        if 'next_obs' not in kwargs:
            raise KeyError("Missing required argument: 'next_obs'")

        state = kwargs['state']
        action = kwargs['action']
        next_obs = kwargs['next_obs']

        state = np.expand_dims(state, axis=0)
        action = np.expand_dims(action, axis=0)

        # GVF Predictions
        state = utils.tensor(state, device)
        action = utils.tensor(action, device)
        curr_values = self.gvfs.gvf.get_q([state], [action], with_grad=False)
        curr_values = utils.to_np(curr_values)
        # The GVF predicts the expected full return. Need to take into account that we're neglecting some percentage of the return.
        corrected_values = curr_values * (1.0 - self.ret_perc)
        self.values.appendleft(corrected_values)

        curr_cumulants = next_obs[self.endo_inds]

        # Update Partial GVF Returns
        self.update_partial_returns(curr_cumulants)

        # Detect alerts
        info = self.initialize_alert_info()
        if len(self.partial_returns) == self.return_steps:
            abs_diffs = abs(self.partial_returns[-1] - self.values[-1])
            self.alert_trace = ((1.0 - self.trace_decay) * abs_diffs) + (self.trace_decay * self.alert_trace)
            individual_alerts = (self.alert_trace > self.trace_thresh).tolist()

            for i in range(len(self.endo_obs_names)):
                cumulant_name = self.endo_obs_names[i]
                info["alert_trace"][self.alert_type()][cumulant_name].append(
                    self.alert_trace[i].squeeze().astype(float).item())
                info["alert"][self.alert_type()][cumulant_name].append(individual_alerts[i])
                info["value"][self.alert_type()][cumulant_name].append(self.values[-1][i].squeeze().astype(float).item())
                info["return"][self.alert_type()][cumulant_name].append(
                    self.partial_returns[-1][i].squeeze().astype(float).item())

        return info

    def get_dim(self) -> int:
        return self.num_gvfs

    def get_discount_factors(self) -> list[float]:
        return [self.gamma]

    def alert_type(self) -> str:
        return "gvf"

    def get_cumulant_names(self) -> list[str]:
        return self.endo_obs_names

    def get_cumulants(self, **kwargs) -> list[float]:
        if 'obs' not in kwargs:
            raise KeyError("Missing required argument: 'obs'")

        return list(kwargs['obs'][self.endo_inds])

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

    def get_test_state_qs(self, plot_info, repeated_test_states, repeated_actions, num_states, test_actions):
        q_values, ensemble_qs = self.gvfs.gvf.get_qs([repeated_test_states], [repeated_actions], with_grad=False)
        q_values = utils.to_np(q_values)
        q_values = q_values.reshape(num_states, test_actions, self.get_dim())
        ensemble_qs = utils.to_np(ensemble_qs)
        ensemble_qs = ensemble_qs.reshape(self.gvfs.gvf.model.ensemble, num_states, test_actions, self.get_dim())

        cumulant_names = self.get_cumulant_names()
        for i in range(len(cumulant_names)):
            cumulant_name = cumulant_names[i]
            plot_info["q_values"][cumulant_name] = q_values[:,:,i]
            plot_info["ensemble_qs"][cumulant_name] = ensemble_qs[:,:,:,i]

        return plot_info

class GVFUncertaintyAlert(GVFTraceAlert):
    def __init__(self, cfg: DictConfig, cumulant_start_ind: int, **kwargs):
        super().__init__(cfg, cumulant_start_ind, **kwargs)

        self.std_thresh = cfg.std_thresh

        self.means = deque([], self.return_steps)
        self.stds = deque([], self.return_steps)

        self.active_alert = [False for i in range(self.get_dim())]

    def evaluate(self, **kwargs) -> dict:
        if 'state' not in kwargs:
            raise KeyError("Missing required argument: 'state'")
        if 'action' not in kwargs:
            raise KeyError("Missing required argument: 'action'")
        if 'next_obs' not in kwargs:
            raise KeyError("Missing required argument: 'next_obs'")

        state = kwargs['state']
        action = kwargs['action']
        next_obs = kwargs['next_obs']

        state = np.expand_dims(state, axis=0)
        action = np.expand_dims(action, axis=0)

        # GVF Predictions
        state = utils.tensor(state, device)
        action = utils.tensor(action, device)
        gvf_qs, gvf_ens = self.gvfs.gvf.get_qs([state], [action], with_grad=False)
        gvf_ens = utils.to_np(gvf_ens)
        gvf_means = gvf_ens.mean(axis=0)
        self.means.appendleft(gvf_means)
        gvf_stds = gvf_ens.std(axis=0)
        print("GVF Alert STDs:", gvf_stds)
        self.stds.appendleft(gvf_stds)
        # The critic predicts the expected full return. Need to take into account that we're neglecting some percentage of the return.
        gvf_qs = utils.to_np(gvf_qs)
        corrected_qs = gvf_qs * (1.0 - self.ret_perc)
        self.values.appendleft(corrected_qs)

        curr_cumulants = next_obs[self.endo_inds]

        # Update Partial GVF Returns
        self.update_partial_returns(curr_cumulants)

        # Detect alerts
        info = self.initialize_alert_info()
        if len(self.partial_returns) == self.return_steps:
            curr_returns = self.partial_returns[-1].astype(float)
            curr_stds = self.stds[-1].astype(float)
            curr_means = self.means[-1].astype(float)
            curr_values = self.values[-1].astype(float)
            for i in range(len(self.endo_obs_names)):
                cumulant_name = self.endo_obs_names[i]
                # info["std"][self.alert_type()][cumulant_name].append(curr_stds[i].item() / abs(curr_means[i].item()))
                info["std"][self.alert_type()][cumulant_name].append(curr_stds[i].item())
                info["return"][self.alert_type()][cumulant_name].append(curr_returns[i].item())
                info["value"][self.alert_type()][cumulant_name].append(curr_values[i].item())

                # Only take into account std of ensemble when alert isn't active
                if self.active_alert[i] or (not self.active_alert[i] and curr_stds[i].item() <= self.std_thresh):
                    abs_diff = abs(curr_returns[i] - curr_values[i])
                    self.alert_trace[i] = ((1.0 - self.trace_decay) * abs_diff) + (self.trace_decay * self.alert_trace[i])
                    self.active_alert[i] = bool(self.alert_trace[i] > self.trace_thresh)
                    info["alert"][self.alert_type()][cumulant_name].append(self.active_alert[i])
                else:
                    # If no alert is active and the std of the ensemble is above the threshold, don't throw an alert
                    abs_diff = 0.0
                    self.alert_trace[i] = ((1.0 - self.trace_decay) * abs_diff) + (self.trace_decay * self.alert_trace[i])
                    self.active_alert[i] = False
                    info["alert"][self.alert_type()][cumulant_name].append(self.active_alert[i])

                info["alert_trace"][self.alert_type()][cumulant_name].append(self.alert_trace[i].item())

        return info

    def initialize_alert_info(self) -> dict:
        info = {}

        info["value"] = {}
        info["return"] = {}
        info["alert_trace"] = {}
        info["alert"] = {}
        info["std"] = {}

        for key in info:
            info[key][self.alert_type()] = {}
            for cumulant_name in self.get_cumulant_names():
                info[key][self.alert_type()][cumulant_name] = []

        return info

    def get_std_thresh(self) -> dict:
        std_threshes = {}
        std_threshes[self.alert_type()] = {}
        for cumulant_name in self.get_cumulant_names():
            std_threshes[self.alert_type()][cumulant_name] = self.std_thresh

        return std_threshes
