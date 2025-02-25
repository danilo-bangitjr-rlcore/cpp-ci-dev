from pathlib import Path

from pydantic import Field

from corerl.agent.greedy_ac import GreedyACConfig
from corerl.configs.config import MISSING, config, post_processor
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.data_pipeline.transforms import AddRawConfig, BoundsConfig, DeltaConfig, NormalizerConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.eval.config import EvalConfig
from corerl.eval.data_report import ReportConfig
from corerl.eval.evals import EvalDBConfig
from corerl.eval.metrics import MetricsConfig, MetricsDBConfig
from corerl.experiment.config import ExperimentConfig
from corerl.interaction.factory import InteractionConfig
from corerl.messages.factory import EventBusConfig


@config()
class DBConfig:
    drivername: str = 'postgresql+psycopg2'
    username: str = 'postgres'
    password: str = 'password'
    ip: str = 'localhost'
    port: int = 5432
    db_name: str = 'postgres'


@config()
class InfraConfig:
    db: DBConfig = Field(default_factory=DBConfig)


@config()
class FeatureFlags:
    delta_actions: bool = False


@config()
class MainConfig:
    # --------------------
    # -- Infrastructure --
    # --------------------
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsDBConfig)
    evals: EvalDBConfig = Field(default_factory=EvalDBConfig, discriminator='name')
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    log_path: Path | None = None

    # -----------
    # -- Agent --
    # -----------
    env: AsyncEnvConfig = MISSING
    interaction: InteractionConfig = MISSING
    agent: GreedyACConfig = Field(default_factory=GreedyACConfig, discriminator='name')

    # ----------------
    # -- Evaluation --
    # ----------------
    eval_cfgs: EvalConfig = Field(default_factory=EvalConfig)
    report : ReportConfig = Field(default_factory=ReportConfig)

    # ---------------
    # -- Computeds --
    # ---------------
    @post_processor
    def _enable_delta_actions(self, cfg: 'MainConfig'):
        if not self.feature_flags.delta_actions:
            assert self.agent.delta_action is False, \
                'delta_actions is disabled but agent is configured to use delta actions'

        self.agent.delta_action = self.feature_flags.delta_actions

        sorted_tags = sorted(self.pipeline.tags, key=lambda x: x.name)
        for tag in sorted_tags:
            if tag.change_bounds is None:
                continue

            self.agent.delta_bounds.append(tag.change_bounds)

            if tag.action_constructor is None:
                tag.action_constructor = []

            tag.action_constructor = [
                DeltaConfig(),
                BoundsConfig(bounds=tag.change_bounds, mode='nan'),
                NormalizerConfig(min=tag.change_bounds[0], max=tag.change_bounds[1]),
                AddRawConfig(),
            ] + tag.action_constructor
