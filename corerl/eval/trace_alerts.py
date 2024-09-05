from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class TraceAlertsEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        if 'alerts' not in kwargs:
            raise KeyError("Missing required argument: 'alerts'")

        self.alerts = kwargs['alerts']
        alerts = self.alerts.get_alerts()
        self.composite_alerts = []
        self.individual_alerts = {}
        self.alert_traces = {}
        self.values = {}
        self.returns = {}
        for alert in alerts:
            alert_type = alert.alert_type()
            self.individual_alerts[alert_type] = {}
            self.alert_traces[alert_type] = {}
            self.values[alert_type] = {}
            self.returns[alert_type] = {}
            for cumulant_name in alert.get_cumulant_names():
                self.individual_alerts[alert_type][cumulant_name] = []
                self.alert_traces[alert_type][cumulant_name] = []
                self.values[alert_type][cumulant_name] = []
                self.returns[alert_type][cumulant_name] = []

        self.alert_trace_thresholds = self.alerts.get_trace_thresh()

    def do_eval(self, **kwargs) -> None:
        if 'alert_info_list' not in kwargs:
            raise KeyError("Missing required argument: 'alert_info_list'")

        alert_info_list = kwargs['alert_info_list']
        for alert_info in alert_info_list:
            self.composite_alerts += alert_info["composite_alert"]
            for alert_type in self.individual_alerts:
                for cumulant_name in self.individual_alerts[alert_type]:
                    self.individual_alerts[alert_type][cumulant_name] += alert_info["alert"][alert_type][cumulant_name]
                    self.alert_traces[alert_type][cumulant_name] += alert_info["alert_trace"][alert_type][cumulant_name]
                    self.values[alert_type][cumulant_name] += alert_info["value"][alert_type][cumulant_name]
                    self.returns[alert_type][cumulant_name] += alert_info["return"][alert_type][cumulant_name]

    def get_stats(self):
        stats = {}

        stats["composite_alerts"] = self.composite_alerts
        stats["individual_alerts"] = self.individual_alerts
        stats["alert_traces"] = self.alert_traces
        stats["alert_values"] = self.values
        stats["alert_returns"] = self.returns
        stats["alert_trace_thresholds"] = self.alert_trace_thresholds

        return stats

    def output(self, path: Path):
        stats = self.get_stats()
        return stats

class UncertaintyAlertsEval(TraceAlertsEval):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        alerts = self.alerts.get_alerts()
        self.std_traces = {}
        for alert in alerts:
            alert_type = alert.alert_type()
            self.std_traces[alert_type] = {}
            for cumulant_name in alert.get_cumulant_names():
                self.std_traces[alert_type][cumulant_name] = []

        self.std_trace_thresholds = self.alerts.get_std_thresh()

    def do_eval(self, **kwargs) -> None:
        if 'alert_info_list' not in kwargs:
            raise KeyError("Missing required argument: 'alert_info_list'")

        alert_info_list = kwargs['alert_info_list']

        super().do_eval(**kwargs)

        for alert_info in alert_info_list:
            for alert_type in self.individual_alerts:
                for cumulant_name in self.individual_alerts[alert_type]:
                    self.std_traces[alert_type][cumulant_name] += alert_info["std_trace"][alert_type][cumulant_name]

    def get_stats(self):
        stats = super().get_stats()

        stats["std_traces"] = self.std_traces
        stats["std_trace_thresholds"] = self.std_trace_thresholds

        return stats




