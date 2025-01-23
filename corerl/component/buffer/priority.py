import logging
from collections.abc import Sequence

import numpy as np
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from discrete_dists.uniform import Uniform

from corerl.component.buffer.base import BaseReplayBufferConfig, ReplayBuffer, buffer_group
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition

logger = logging.getLogger(__name__)


@config()
class PriorityReplayBufferConfig(BaseReplayBufferConfig):
    name: str = "priority"

    uniform_probability: float = 0.01
    priority_decay: float = 0.99


class PriorityBuffer(ReplayBuffer):
    def __init__(self, cfg: PriorityReplayBufferConfig):
        super().__init__(cfg)

        self._idx_dist = MixtureDistribution([
            # build a uniform distribution with no support (support grows
            # as elements are added to the buffer)
            SubDistribution(d=Uniform(0), p=cfg.uniform_probability),
            # build a distribution that samples proportionally to the
            # priorities added to it.
            SubDistribution(d=Proportional(self.memory), p=1-cfg.uniform_probability),
        ])

        self._max_priority = 1

    def _sample_indices(self):
        return self._idx_dist.stratified_sample(self.rng, self.batch_size)


    def feed(self, transitions: Sequence[Transition]):
        idxs = super().feed(transitions)
        priorities = np.ones(len(idxs)) * self._max_priority
        self._idx_dist.update(idxs, priorities)

        return idxs


    def load(self, transitions: Sequence[Transition]):
        idxs = super().load(transitions)
        priorities = np.ones(len(idxs)) * self._max_priority
        self._idx_dist.update(idxs, priorities)

        return idxs


    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray):
        self._idx_dist.update(idxs, priorities)


buffer_group.dispatcher(PriorityBuffer)
