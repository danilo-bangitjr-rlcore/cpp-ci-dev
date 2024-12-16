from dataclasses import field
from typing import Any

from corerl.agent import AgentConfig
from corerl.configs.config import config
from corerl.environment.async_env.sim_async_env import SimAsyncEnvConfig
from corerl.experiment.config import ExperimentConfig
from corerl.data_pipeline.pipeline import PipelineConfig


@config()
class MainConfig:
    action_period: int
    obs_period: int
    agent: AgentConfig
    alerts: Any
    env: SimAsyncEnvConfig

    use_alerts: bool = False
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
