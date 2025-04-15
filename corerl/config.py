from pathlib import Path

from pydantic import Field

from corerl.agent.greedy_ac import GreedyACConfig
from corerl.configs.config import config, post_processor
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval.config import EvalConfig
from corerl.eval.data_report import ReportConfig
from corerl.eval.evals import EvalDBConfig
from corerl.eval.metrics import MetricsDBConfig
from corerl.experiment.config import ExperimentConfig
from corerl.interaction.factory import InteractionConfig
from corerl.messages.factory import EventBusConfig


@config()
class DBConfig:
    """
    Kind: optional external

    Default configurations for our timeseries
    database. Should generally be owned by RLCore
    during setup, however in more extreme circumstances
    this may be owned by the user on an internal network.
    """
    drivername: str = 'postgresql+psycopg2'
    username: str = 'postgres'
    password: str = 'password'
    ip: str = 'localhost'
    port: int = 5432
    db_name: str = 'postgres'
    schema: str = 'public'


@config()
class InfraConfig:
    """
    Kind: optional external

    Infrastructure configuration for wiring the agent
    into the external system.
    """
    db: DBConfig = Field(default_factory=DBConfig)


@config()
class FeatureFlags:
    """
    Kind: internal

    Flags to enable/disable new features in the agent.
    These should default to False and the date that they
    are added should be recorded.

    Feature flags should generally be for internal use only.
    Their impact on the customer cannot be easily guaranteed
    or communicated.

    See documentation:
    https://docs.google.com/document/d/1Inm7dMHIRvIGvM7KByrRhxHsV7uCIZSNsddPTrqUcOU/edit?tab=t.4238yb3saoju
    """
    # 2025-02-01
    delta_actions: bool = False

    # 2025-02-01
    ensemble: int = 1

    # 2025-03-01
    zone_violations: bool = False

    # 2025-03-13
    action_embedding: bool = False

    # 2025-03-24
    wide_nets: bool = True

    # 2025-04-14
    action_bounds: bool = False

@config()
class MainConfig:
    """
    Top-level configuration for corerl.
    This contains a mix of internal and external configurations.
    """
    # --------------------
    # -- Infrastructure --
    # --------------------
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    log_path: Path | None = None

    evals: EvalDBConfig = Field(default_factory=EvalDBConfig)
    metrics: MetricsDBConfig = Field(default_factory=MetricsDBConfig)
    """
    Kind: internal

    Metrics logging location. By default points to the timeseries
    database. Optionally can point to a local csv file.
    """

    # -----------
    # -- Agent --
    # -----------
    env: AsyncEnvConfig = Field(default_factory=AsyncEnvConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
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
        assert not self.feature_flags.action_bounds, "Behavior of delta actions + action bounds undefined"
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
        self.agent.critic.critic_network.base.input_scales = [
            # state
            0.75,
            # action
            0.25,
        ]

        # remove the first layer of the combined network
        # split it in half and assign one half to each input
        hidden = self.agent.critic.critic_network.base.combined_cfg.hidden.pop(0)
        hidden_act = self.agent.critic.critic_network.base.combined_cfg.activation.pop(0)
        self.agent.critic.critic_network.base.input_cfg.hidden = [int(hidden // 2)]
        self.agent.critic.critic_network.base.input_cfg.activation = [hidden_act]
