from typing import Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from collections.abc import MutableMapping
from hydra.core.config_store import ConfigStore

from corerl.experiment.config import ExperimentConfig

@dataclass
class MainConfig(MutableMapping):
    use_alerts: bool = False
    agent: Any = MISSING
    agent_transition_creator: Any = MISSING
    alert_transition_creator: Any = MISSING
    alerts: Any = MISSING
    calibration_model: Any = MISSING
    data_loader: Any = MISSING
    env: Any = MISSING
    eval: Any = MISSING
    event_bus: Any = MISSING
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    interaction: Any = MISSING
    normalizer: Any = MISSING
    offline_data: Any = MISSING
    state_constructor: Any = MISSING

cs = ConfigStore.instance()
cs.store(name='base_config', node=MainConfig)
