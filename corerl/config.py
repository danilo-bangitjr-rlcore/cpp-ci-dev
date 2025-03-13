from pathlib import Path

from pydantic import Field

from corerl.agent.greedy_ac import GreedyACConfig
from corerl.configs.config import MISSING, config, post_processor
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
    # 2025-02-01
    delta_actions: bool = False

    # 2025-02-01
    ensemble: int = 1

    # 2025-03-01
    zone_violations: bool = False

    # 2025-03-13
    action_embedding: bool = False


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
            assert self.agent.policy.delta_actions is False, \
                'delta_actions is disabled but actor is configured to use delta actions'

        self.agent.policy.delta_actions = self.feature_flags.delta_actions

        if not self.feature_flags.delta_actions:
            return

        sorted_tags = sorted(self.pipeline.tags, key=lambda x: x.name)
        for tag in sorted_tags:
            # if not an action, continue
            if tag.change_bounds is None:
                continue

            if tag.action_constructor is None:
                tag.action_constructor = []

            self.agent.policy.delta_bounds.append(tag.change_bounds)

        for tag in sorted_tags:
            # if not an action, continue

            if tag.action_constructor is not None:
                assert tag.change_bounds is not None, \
                    f"Delta actions enabled, but change bounds for tag {tag.name} unspecified."


    @post_processor
    def _enable_ensemble(self, cfg: 'MainConfig'):
        ensemble_size = self.feature_flags.ensemble
        self.agent.critic.critic_network.ensemble = ensemble_size
        self.agent.critic.buffer.ensemble = ensemble_size

        if ensemble_size == 1:
            self.agent.critic.buffer.ensemble_probability = 1.


    @post_processor
    def _enable_action_embedding(self, cfg: 'MainConfig'):
        if not self.feature_flags.action_embedding:
            return

        self.agent.critic.critic_network.base.skip_input = False

        # remove the first layer of the combined network
        # split it in half and assign one half to each input
        hidden = self.agent.critic.critic_network.base.combined_cfg.hidden.pop(0)
        self.agent.critic.critic_network.base.input_cfg.hidden = [int(hidden // 2)]
