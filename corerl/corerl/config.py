import json
from pathlib import Path

import yaml
from coreio.config import CoreIOConfig
from lib_config.config import MISSING, computed, config, post_processor
from lib_config.loader import config_to_json
from lib_defs.config_defs.tag_config import TagType
from pydantic import Field

from corerl.configs.agent.greedy_ac import GreedyACConfig
from corerl.configs.data_pipeline.pipeline_config import PipelineConfig
from corerl.configs.environment.async_env import AsyncEnvConfig
from corerl.configs.eval.config import EvalConfig
from corerl.configs.eval.evals import EvalDBConfig
from corerl.configs.eval.metrics import MetricsDBConfig
from corerl.configs.infra import FeatureFlags, InfraConfig
from corerl.configs.interaction.config import InteractionConfig
from corerl.configs.messages.event_bus import EventBusConfig
from corerl.configs.messages.event_bus_client import EventBusClientConfig


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
    event_bus_client: EventBusClientConfig = Field(default_factory=EventBusClientConfig)
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
    demo_mode: bool = False

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
