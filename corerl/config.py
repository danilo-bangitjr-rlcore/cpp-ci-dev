from dataclasses import field
from typing import Any

from pydantic import Field

from corerl.agent import AgentConfig
from corerl.agent.random import RandomAgentConfig
from corerl.configs.config import MISSING, config
from corerl.environment.async_env.sim_async_env import SimAsyncEnvConfig
from corerl.experiment.config import ExperimentConfig
from corerl.data_pipeline.pipeline import PipelineConfig


@config()
class MainConfig:
    action_period: int = MISSING
    obs_period: int = MISSING
    alerts: Any = None
    use_alerts: bool = False
    env: SimAsyncEnvConfig = field(default_factory=SimAsyncEnvConfig)

    agent: AgentConfig = Field(default_factory=RandomAgentConfig, discriminator='name')
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
