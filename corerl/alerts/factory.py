from omegaconf import DictConfig
from corerl.alerts.base import BaseAlert
from corerl.alerts.action_value import ActionValueTraceAlert, ActionValueUncertaintyAlert
from corerl.alerts.gvf import GVFTraceAlert, GVFUncertaintyAlert

def init_alert(cfg: DictConfig, cumulant_start_ind: int, alert_args: dict) -> BaseAlert:
    """
    config files: corerl/config/alerts
    """
    name = cfg.name

    if name == "action_value_trace":
        return ActionValueTraceAlert(cfg, cumulant_start_ind, **alert_args)
    elif name == "action_value_uncertainty":
        return ActionValueUncertaintyAlert(cfg, cumulant_start_ind, **alert_args)
    elif name == "gvf_trace":
        return GVFTraceAlert(cfg, cumulant_start_ind, **alert_args)
    elif name == "gvf_uncertainty":
        return GVFUncertaintyAlert(cfg, cumulant_start_ind, **alert_args)
    else:
        raise NotImplementedError
