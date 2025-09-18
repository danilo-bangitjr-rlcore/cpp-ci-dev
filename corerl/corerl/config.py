import json
from pathlib import Path

import yaml
from coreio.config import CoreIOConfig
from lib_config.config import MISSING, computed, config, post_processor
from lib_config.loader import config_to_json
from lib_defs.config_defs.tag_config import TagType
from pydantic import Field

from corerl.agent.greedy_ac import GreedyACConfig
from corerl.data_pipeline.pipeline_config import PipelineConfig
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval.config import EvalConfig
from corerl.eval.evals.base import EvalDBConfig
from corerl.eval.metrics.base import MetricsDBConfig
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
    ensemble: int = 2

    # 2025-04-29
    recency_bias_buffer: bool = False

    # 2025-05-04
    interaction_action_variance: bool = False

    # 2025-05-14
    regenerative_optimism: bool = False

    # 2025-05-26
    normalize_return: bool = False

    # 2025-06-06
    autoencoder_imputer: bool = True

    # 2025-06-11
    nominal_setpoint_bias: bool = True

    # 2025-06-22
    noisy_networks: bool = False

    # 2025-06-27
    higher_critic_lr: bool = True

    # 2025-06-27
    mu_sigma_multipliers: bool = False

    # 2025-07-24
    wide_metrics: bool = False

    # 2025-08-15
    state_layer_norm: bool = True


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
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    save_path: Path = MISSING
    log_path: Path | None = None
    silent: bool = False

    evals: EvalDBConfig = Field(default_factory=EvalDBConfig)
    metrics: MetricsDBConfig = Field(default_factory=MetricsDBConfig)

    """
    Top-level configuration for coreio
    """
    coreio: CoreIOConfig = Field(default_factory=CoreIOConfig)

    """
    Kind: internal

    Metrics logging location. By default points to the timeseries
    database. Optionally can point to a local csv file.
    """

    # -------------
    # -- Problem --
    # -------------
    max_steps: int | None = None
    seed: int = 0  # affects agent and env
    is_simulation: bool = True

    # -----------
    # -- Agent --
    # -----------
    agent_name: str = 'corey'  # typically should indicate the process the agent is controlling
    env: AsyncEnvConfig = Field(default_factory=AsyncEnvConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    agent: GreedyACConfig = Field(default_factory=GreedyACConfig, discriminator='name')

    # ----------------
    # -- Evaluation --
    # ----------------
    eval_cfgs: EvalConfig = Field(default_factory=EvalConfig)

    # ---------------
    # -- Computeds --
    # ---------------
    @post_processor
    def _enable_ensemble(self, cfg: 'MainConfig'):
        ensemble_size = self.feature_flags.ensemble
        if ensemble_size == 1:
            self.agent.critic.buffer.ensemble_probability = 1.

    @post_processor
    def _enable_higher_critic_lr(self, cfg: 'MainConfig'):
        if self.feature_flags.higher_critic_lr:
            self.agent.critic.stepsize = 0.001

    @post_processor
    def _enable_mu_sigma_multipliers(self, cfg: 'MainConfig'):
        if self.feature_flags.mu_sigma_multipliers:
            self.agent.policy.mu_multiplier = 10.0
            self.agent.policy.sigma_multiplier = 1.0

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
            if tag_cfg.type == TagType.ai_setpoint and tag_cfg.guardrail_schedule is not None:
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
    def _regenerative_optimism(self, cfg: 'MainConfig'):
        if not self.feature_flags.regenerative_optimism:
            return

        self.agent.policy.sort_noise = 0.02
        self.agent.critic.action_regularization = 0.0001

        if self.feature_flags.normalize_return:
            self.agent.policy.sort_noise *= (1 - self.agent.gamma)

    @post_processor
    def _enable_return_normalization(self, cfg: 'MainConfig'):
        if not self.feature_flags.normalize_return:
            return

        self.agent.loss_threshold = 1e-8

    @post_processor
    def _enable_wide_metrics(self, cfg: 'MainConfig'):
        if not self.feature_flags.wide_metrics:
            return

        self.metrics.narrow_format = False
        self.metrics.table_name = self.metrics.table_name + '_wide'
