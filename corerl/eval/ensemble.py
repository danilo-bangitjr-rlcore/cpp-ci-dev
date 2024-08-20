from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json
import numpy as np


class EnsembleEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        if 'alerts' not in kwargs:
            raise KeyError("Missing required argument: 'alerts'")

        self.alerts = kwargs['alerts']
        alerts = self.alerts.get_alerts()
        self.avg_stds = {}
        for alert in alerts:
            alert_type = alert.alert_type()
            self.avg_stds[alert_type] = {}
            for cumulant_name in alert.get_cumulant_names():
                self.avg_stds[alert_type][cumulant_name] = []

    def do_eval(self, **kwargs) -> None:
        if 'ensemble_info' not in kwargs:
            raise KeyError("Missing required argument: 'ensemble_info'")

        ensemble_info = kwargs['ensemble_info']

        if len(ensemble_info) > 0:
            for alert_type in ensemble_info:
                for cumulant_name in ensemble_info[alert_type]:
                    step_avg_std = ensemble_info[alert_type][cumulant_name]["std"].mean()
                    self.avg_stds[alert_type][cumulant_name].append(step_avg_std)

    def get_stats(self):
        stats = {}
        stats["training_stds"] = self.avg_stds
        return stats

    def output(self, path: Path):
        stats = self.get_stats()
        return stats
