from typing import Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from corerl.data.base_tc import BaseTCConfig
from corerl.experiment.config import ExperimentConfig
from corerl.data.normalizer.base import NormalizerConfig
from corerl.data_pipeline.base import BaseDataLoaderConfig
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.interaction.anytime_interaction import AnytimeInteractionConfig
from corerl.interaction.base import BaseInteractionConfig
from corerl.state_constructor.base import SCConfig

@dataclass
class MainConfig:
    use_alerts: bool = False
    agent: Any = MISSING
    agent_transition_creator: Any = MISSING
    alert_transition_creator: BaseTCConfig | None = MISSING
    alerts: Any = MISSING
    calibration_model: Any = MISSING
    data_loader: BaseDataLoaderConfig = MISSING
    env: Any = MISSING
    eval: Any = MISSING
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    interaction: BaseInteractionConfig = field(default_factory=AnytimeInteractionConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    offline_data: Any = MISSING
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    state_constructor: SCConfig = MISSING


cs = ConfigStore.instance()
cs.store(name='base_config', node=MainConfig)
