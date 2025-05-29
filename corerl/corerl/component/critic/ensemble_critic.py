import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field, TypeAdapter

from corerl.component.buffer import BufferConfig, MixedHistoryBufferConfig, RecencyBiasBufferConfig
from corerl.component.network.networks import EnsembleNetwork, EnsembleNetworkConfig
from corerl.component.optimizers.factory import OptimizerConfig
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.datatypes import TransitionBatch
from corerl.state import AppState

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)

@config()
class CriticConfig:
    """
    Kind: internal

    Critic-specific hyperparameters.
    """
    action_regularization: float = 0.0
    num_rand_actions: int = 10
    critic_network: EnsembleNetworkConfig = Field(default_factory=EnsembleNetworkConfig)
    critic_optimizer: OptimizerConfig = Field(default_factory=AdamConfig)
    buffer: BufferConfig = MISSING
    grad_clip: float = 50_000

    @computed('buffer')
    @classmethod
    def _buffer(cls, cfg: 'MainConfig'):
        default_buffer_type = (
            RecencyBiasBufferConfig
            if cfg.feature_flags.recency_bias_buffer else
            MixedHistoryBufferConfig
        )

        ta = TypeAdapter(default_buffer_type)
        default_buffer = default_buffer_type(id='critic')
        default_buffer_dict = ta.dump_python(default_buffer, warnings=False)
        main_cfg: Any = cfg
        out = ta.validate_python(default_buffer_dict, context=main_cfg)
        return out


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: CriticConfig, app_state: AppState):
        self._app_state = app_state
        self._cfg = cfg

    @abstractmethod
    def update(
        self,
        batches: list[TransitionBatch],
        next_actions: list[torch.Tensor],
        eval_batches: list[TransitionBatch],
        eval_actions: list[torch.Tensor],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------- #
#                                    Metrics                                   #
# ---------------------------------------------------------------------------- #

def log_critic_gradient_norm(app_state: AppState, critic: EnsembleNetwork):
    """
    Logs the gradient norm for each member of the ensemble.
    """
    for i, norm in enumerate(critic.get_gradient_norms()):
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric=f"optimizer_critic_{i}_grad_norm",
            value=norm,
        )

def log_critic_weight_norm(app_state: AppState, critic: EnsembleNetwork):
    """
    Logs the weight norm for each member of the ensemble.
    """
    with torch.no_grad():
        for i, norm in enumerate(critic.get_weight_norms()):
            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"network_critic_{i}_weight_norm",
                value=norm,
            )

def log_critic_stable_rank(app_state: AppState, critic: EnsembleNetwork):
    with torch.no_grad():
        for i, stable_ranks in enumerate(critic.get_stable_ranks()):
            for j, rank in enumerate(stable_ranks):
                app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric=f"network_critic_{i}_stable_rank_layer_{j}",
                    value=rank,
                )
