from dataclasses import field

from pydantic import Field

from corerl.agent import AgentConfig
from corerl.agent.random import RandomAgentConfig
from corerl.configs.config import MISSING, config
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.eval.writer import MetricsConfig, MetricsDBConfig
from corerl.experiment.config import ExperimentConfig
from corerl.interaction.factory import InteractionConfig


@config()
class MainConfig:
    interaction: InteractionConfig = MISSING
    metrics: MetricsConfig = Field(default_factory=MetricsDBConfig, discriminator='name')

    env: AsyncEnvConfig = MISSING # field(default_factory=SimAsyncEnvConfig)
    agent: AgentConfig = Field(default_factory=RandomAgentConfig, discriminator='name')
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
