from omegaconf import DictConfig

from corerl.alerts.base import BaseAlert
from corerl.alerts.factory import init_alert
from corerl.data.data import Transition


class CompositeAlert(BaseAlert):
    def __init__(self, cfg: DictConfig, alert_args: dict):
        self.alerts = []
        if cfg:
            for alert_type in cfg.keys():
                alert_type_cfg = cfg[alert_type]
                self.alerts.append(init_alert(alert_type_cfg, alert_args))

    def update_buffer(self, transitions: list[Transition]) -> None:
        assert len(transitions) == len(self.alerts), "Need a new transition for each type of alert"
        for i in range(len(transitions)):
            self.alerts[i].update_buffer(transitions[i])

    def update(self) -> None:
        for alert in self.alerts:
            alert.update()

    def evaluate(self, **kwargs) -> dict:
        alerts_info = {}
        alerts_info["composite_alert"] = [False]
        for alert in self.alerts:
            alert_info = alert.evaluate(**kwargs)
            for key in alert_info:
                if key not in alerts_info:
                    alerts_info[key] = {}
                for sensor_name in alert_info[key]:
                    alerts_info[key][sensor_name] = alert_info[key][sensor_name]
                    if key == "alert":
                        alerts_info["composite_alert"][0] |= alert_info[key][sensor_name][0]

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
            for key in alert_trace_threshes:
                trace_threshes[key] = alert_trace_threshes[key]

        return trace_threshes
