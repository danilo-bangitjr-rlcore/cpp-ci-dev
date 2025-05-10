import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from pydantic import Field

from corerl.agent.greedy_ac import GreedyACConfig
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.configs.config import MISSING, computed, config, list_, post_processor
from corerl.configs.loader import config_to_json
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval.config import EvalConfig
from corerl.eval.data_report import ReportConfig
from corerl.eval.evals import EvalDBConfig
from corerl.eval.metrics import MetricsDBConfig
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
    device: str = 'cpu'
    num_threads: int = 4


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
    ensemble: int = 1

    # 2025-03-01
    zone_violations: bool = False

    # 2025-03-13
    action_embedding: bool = True

    # 2025-03-24
    wide_nets: bool = True

    # 2025-04-14
    action_bounds: bool = False

    # 2025-04-25
    use_residual: bool = False

    # 2025-04-29
    recency_bias_buffer: bool = False

    # 2025-04-28
    prod_265_ignore_oob_tags_in_compound_goals: bool = False

    # 2025-05-04
    interaction_action_variance: bool = False


@config()
class OfflineConfig:
    offline_steps: int = 0
    offline_eval_iters: list[int] = list_()
    offline_start_time: datetime | None = None
    offline_end_time: datetime | None = None
    pipeline_batch_duration: timedelta = timedelta(days=7)


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
    offline: OfflineConfig = Field(default_factory=OfflineConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    save_path: Path = MISSING
    log_path: Path | None = None
    silent: bool = False

    evals: EvalDBConfig = Field(default_factory=EvalDBConfig)
    metrics: MetricsDBConfig = Field(default_factory=MetricsDBConfig)
    """
    Kind: internal

    Metrics logging location. By default points to the timeseries
    database. Optionally can point to a local csv file.
    """

    # -------------
    # -- Problem --
    # -------------
    max_steps: int | None = None
    seed: int = 0 # affects agent and env
    is_simulation: bool = True

    # -----------
    # -- Agent --
    # -----------
    agent_name: str = 'corey' # typically should indicate the process the agent is controlling
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
    def _enable_ensemble(self, cfg: 'MainConfig'):
        ensemble_size = self.feature_flags.ensemble
        self.agent.critic.critic_network.ensemble = ensemble_size
        self.agent.critic.buffer.ensemble = ensemble_size

        if ensemble_size == 1:
            self.agent.critic.buffer.ensemble_probability = 1.

    @post_processor
    def _enable_recency_bias_buffer(self, cfg: 'MainConfig'):
        if not self.feature_flags.recency_bias_buffer:
            return

        assert self.agent.critic.buffer.name == 'recency_bias_buffer'
        assert self.agent.policy.buffer.name == 'recency_bias_buffer'

    @post_processor
    def _enable_time_dilation(self, cfg: 'MainConfig'):
        """
        Divides time-based configs by time_dilation in order to easily experiment
        with running deployment interactions faster.
        """
        self.interaction.obs_period /= self.interaction.time_dilation
        self.interaction.action_period /= self.interaction.time_dilation
        self.interaction.state_age_tol /= self.interaction.time_dilation
        self.interaction.update_period /= self.interaction.time_dilation

        if self.interaction.warmup_period is not None:
            self.interaction.warmup_period /= self.interaction.time_dilation

        self.interaction.checkpoint_freq /= self.interaction.time_dilation
        self.interaction.checkpoint_cliff /= self.interaction.time_dilation
        self.interaction.heartbeat.heartbeat_period /= self.interaction.time_dilation

        for tag_cfg in self.pipeline.tags:
            if tag_cfg.guardrail_schedule is not None:
                tag_cfg.guardrail_schedule.duration /= self.interaction.time_dilation


    @computed('save_path')
    @classmethod
    def _save_path(cls, cfg: 'MainConfig'):
        save_path = (
                Path('outputs') /
                cfg.agent_name /
                (f'seed-{cfg.seed}')
        )

        cfg_json = config_to_json(MainConfig, cfg)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.yaml", "w") as f:
            yaml.safe_dump(json.loads(cfg_json), f)

        return save_path

    @post_processor
    def _enable_zone_violations(self, cfg: 'MainConfig'):
        if not self.feature_flags.zone_violations:
            return

        self.agent.critic.critic_optimizer = AdamConfig(
            lr=0.0001,
            weight_decay=0.001,
        )
        self.agent.policy.optimizer = AdamConfig(
            lr=0.0001,
            weight_decay=0.001,
        )

        self.agent.max_critic_updates = 10
        self.agent.policy.prop_percentile_learned = 0.9
