from dataclasses import field
from pathlib import Path

from pydantic import Field

from corerl.agent import AgentConfig
from corerl.agent.random import RandomAgentConfig
from corerl.configs.config import MISSING, config
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.eval.config import EvalConfig
from corerl.eval.data_report import ReportConfig
from corerl.eval.writer import MetricsConfig, MetricsDBConfig
from corerl.experiment.config import ExperimentConfig
from corerl.interaction.factory import InteractionConfig
from corerl.messages.factory import EventBusConfig


@config()
class MainConfig:
    interaction: InteractionConfig = MISSING
    metrics: MetricsConfig = Field(default_factory=MetricsDBConfig, discriminator='name')
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    env: AsyncEnvConfig = MISSING # field(default_factory=SimAsyncEnvConfig)
    agent: AgentConfig = Field(default_factory=RandomAgentConfig, discriminator='name')
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    report : ReportConfig = field(default_factory=ReportConfig)
    log_path: Path | None = None
