from omegaconf import DictConfig

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

    def update(self) -> None:
        for alert in self.alerts:
            alert.update()

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
                            alerts_info["composite_alert"][0] |= alert_info[field][alert_type][cumulant_name][0]

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

    def get_alerts(self) -> list[BaseAlert]:
        return self.alerts
