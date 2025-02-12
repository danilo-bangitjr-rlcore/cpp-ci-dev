import logging
from collections.abc import Sequence
from typing import Literal

from discrete_dists.uniform import Uniform

from corerl.component.buffer.base import BaseReplayBufferConfig, ReplayBuffer, buffer_group
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import DataMode, Transition
from corerl.state import AppState

logger = logging.getLogger(__name__)


@config()
class UniformReplayBufferConfig(BaseReplayBufferConfig):
    name: Literal['uniform'] = "uniform"


class UniformBuffer(ReplayBuffer):
    def __init__(self, cfg: UniformReplayBufferConfig, app_state: AppState):
        super().__init__(cfg, app_state)

        # initially this dist has no support
        # support expands as samples are added
        self._idx_dist = Uniform(0)

    def _sample_indices(self):
        return self._idx_dist.sample(self.rng, self.batch_size)


    def feed(self, transitions: Sequence[Transition], data_mode: DataMode):
        idxs = super().feed(transitions, data_mode)

        # expand the support of the distribution to cover
        # the entire size of the replay buffer
        self._idx_dist.update_support(self.size[0])

        return idxs


    def load(self, transitions: Sequence[Transition], data_mode: DataMode):
        idxs = super().load(transitions, data_mode)

        # expand the support of the distribution to cover
        # the entire size of the replay buffer
        self._idx_dist.update_support(self.size[0])

        return idxs


    def reset(self):
        super().reset()
        # reset the support to be empty
        self._idx_dist.update_support(0)


buffer_group.dispatcher(UniformBuffer)
