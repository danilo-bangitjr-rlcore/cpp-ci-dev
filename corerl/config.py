from pathlib import Path

from pydantic import Field

from corerl.agent import AgentConfig
from corerl.agent.random import RandomAgentConfig
from corerl.configs.config import MISSING, config
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.eval.config import EvalConfig
from corerl.eval.data_report import ReportConfig
from corerl.eval.evals import EvalDBConfig
from corerl.eval.metrics import MetricsConfig, MetricsDBConfig
from corerl.experiment.config import ExperimentConfig
from corerl.interaction.factory import InteractionConfig
from corerl.messages.factory import EventBusConfig


@config()
class MainConfig:
    # --------------------
    # -- Infrastructure --
    # --------------------
    metrics: MetricsConfig = Field(default_factory=MetricsDBConfig, discriminator='name')
    evals: EvalDBConfig = Field(default_factory=EvalDBConfig, discriminator='name')
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    log_path: Path | None = None

    # -----------
    # -- Agent --
    # -----------
    env: AsyncEnvConfig = MISSING
    interaction: InteractionConfig = MISSING
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    agent: AgentConfig = Field(default_factory=RandomAgentConfig, discriminator='name')

    # ----------------
    # -- Evaluation --
    # ----------------
    eval_cfgs: EvalConfig = Field(default_factory=EvalConfig)
    report : ReportConfig = Field(default_factory=ReportConfig)
