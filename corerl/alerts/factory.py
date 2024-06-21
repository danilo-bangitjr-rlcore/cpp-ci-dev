from omegaconf import DictConfig
from corerl.alerts.base import BaseAlert
from corerl.alerts.action_value import ActionValueAlert
from corerl.alerts.gvf import GVFAlert

def init_alert(cfg: DictConfig, alert_args: dict) -> BaseAlert:
    """
    config files: corerl/config/alerts
    """
    name = cfg.name

    if name == "action_value":
        return ActionValueAlert(cfg, **alert_args)
    elif name == "gvf":
        return GVFAlert(cfg, **alert_args)
    else:
        raise NotImplementedError
