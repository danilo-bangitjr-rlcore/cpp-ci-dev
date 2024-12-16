from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from corerl.experiment.config import ExperimentConfig
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.agent import register as register_agent


@dataclass
class MainConfig:
    action_period: int = MISSING
    obs_period: int = MISSING

    use_alerts: bool = False
    agent: Any = MISSING
    alerts: Any = MISSING
    env: Any = MISSING
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


cs = ConfigStore.instance()
cs.store(name='base_config', node=MainConfig)

register_agent()
