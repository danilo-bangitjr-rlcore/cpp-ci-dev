from omegaconf import DictConfig

import numpy as np
from corerl.component.network.utils import to_np, tensor
from corerl.utils.device import device
from corerl.alerts.base import BaseAlert
from corerl.alerts.factory import init_alert
from corerl.data.data import Transition


class CompositeAlert(BaseAlert):
    def __init__(self, cfg: DictConfig, alert_args: dict):
        self.alerts = []
        if cfg:
            cumulant_start_ind = 0
            for alert_type in cfg.keys():
                alert_type_cfg = cfg[alert_type]
                self.alerts.append(init_alert(alert_type_cfg, cumulant_start_ind, alert_args))
                cumulant_start_ind += self.alerts[-1].get_dim()

    def update_buffer(self, transition: Transition) -> None:
        for alert in self.alerts:
            alert.update_buffer(transition)

    def load_buffer(self, transitions: list[Transition]) -> None:
        for alert in self.alerts:
            alert.load_buffer(transitions)

    def update(self) -> dict:
        composite_ensemble_info = {}
        for alert in self.alerts:
            alert_ensemble_info = alert.update()
            for key in alert_ensemble_info:
                composite_ensemble_info[key] = alert_ensemble_info[key]

        return composite_ensemble_info

    def evaluate(self, **kwargs) -> dict:
        alerts_info = {}
        alerts_info["composite_alert"] = [False]
        for alert in self.alerts:
            alert_info = alert.evaluate(**kwargs)
            for field in alert_info:
                if field not in alerts_info:
                    alerts_info[field] = {}
                for alert_type in alert_info[field]:
                    if alert_type not in alerts_info[field]:
                        alerts_info[field][alert_type] = {}
                    for cumulant_name in alert_info[field][alert_type]:
                        if cumulant_name not in alerts_info[field][alert_type]:
                            alerts_info[field][alert_type][cumulant_name] = {}
                        alerts_info[field][alert_type][cumulant_name] = alert_info[field][alert_type][cumulant_name]
                        if field == "alert" and len(alert_info[field][alert_type][cumulant_name]) == 1:
                            alerts_info["composite_alert"][0] |= bool(alert_info[field][alert_type][cumulant_name][0])

        return alerts_info

    def get_dim(self) -> int:
        dim = 0
        for alert in self.alerts:
            dim += alert.get_dim()

        return dim

    def get_discount_factors(self) -> list[float]:
        gammas = []
        for alert in self.alerts:
            gammas += alert.get_discount_factors()

        return gammas

    def get_cumulants(self, **kwargs) -> list[float]:
        cumulants = []
        for alert in self.alerts:
            cumulants += alert.get_cumulants(**kwargs)

        return cumulants

    def get_trace_thresh(self) -> dict:
        trace_threshes = {}
        for alert in self.alerts:
            alert_trace_threshes = alert.get_trace_thresh()
            alert_type = alert.alert_type()
            trace_threshes[alert_type] = {}
            for cumulant_name in alert_trace_threshes[alert_type]:
                trace_threshes[alert_type][cumulant_name] = alert_trace_threshes[alert_type][cumulant_name]

        return trace_threshes

    def get_std_thresh(self) -> dict:
        std_threshes = {}
        for alert in self.alerts:
            alert_std_threshes = alert.get_std_thresh()
            alert_type = alert.alert_type()
            std_threshes[alert_type] = {}
            for cumulant_name in alert_std_threshes[alert_type]:
                std_threshes[alert_type][cumulant_name] = alert_std_threshes[alert_type][cumulant_name]

        return std_threshes

    def get_alerts(self) -> list[BaseAlert]:
        return self.alerts

    def get_test_state_qs(self, test_transitions):
        test_actions = 100
        num_states = len(test_transitions)
        test_states = []
        for transition in test_transitions:
            test_states.append(transition.state)
        test_states_np = np.array(test_states, dtype=np.float32)

        test_states = tensor(test_states_np, device)
        actions = np.linspace(np.array([0]), np.array([1]), num=test_actions)

        repeated_test_states = test_states.repeat_interleave(test_actions, dim=0)
        repeated_actions = [actions for i in range(num_states)]
        repeated_actions = np.concatenate(repeated_actions)
        repeated_actions = tensor(repeated_actions, device)

        plot_info = {}
        plot_info["states"] = test_states_np
        plot_info["actions"] = actions
        plot_info["q_values"] = {}
        plot_info["ensemble_qs"] = {}

        for alert in self.alerts:
            plot_info = alert.get_test_state_qs(plot_info, repeated_test_states, repeated_actions, num_states, test_actions)

        return plot_info

    def get_buffer_sizes(self):
        for alert in self.alerts:
            alert_type = alert.alert_type()
            print("Get {} Buffer Size(s)".format(alert_type))
            print("{} Buffer Size(s): {}".format(alert_type, alert.get_buffer_size()))

