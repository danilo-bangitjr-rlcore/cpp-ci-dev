from corerl.utils.device import device
from omegaconf import DictConfig
from warnings import warn
import corerl.component.network.utils as network_utils
import numpy as np
import torch
import random

from corerl.component.buffer.buffers import _get_size, _to_tensor


class UniformTrajectoryBuffer:
    def __init__(self, cfg: DictConfig):
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size
        self.data = []
        self.pos = 0
        self.full = False

        if self.batch_size == 0:
            self.sample = self.sample_batch
        else:
            self.sample = self.sample_mini_batch

    def feed(self, trajectory: tuple) -> None:
        self.data.extend(trajectory)

    def sample_mini_batch(self, batch_size: int = None) -> dict:
        total_transitions = self.num_transitions
        probs = [len(i) / total_transitions for i in self.data]
        sampled_data = random.choices(self.data, weights=probs, k=batch_size)
        return sampled_data

    def load(self, trajectories: list) -> None:
        self.data = trajectories

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def num_transitions(self) -> int:
        return sum([len(i) for i in self.data])

    def reset(self) -> None:
        self.data = []
        self.pos = 0

    def get_all_data(self) -> list:
        return self.data

    # def _prepare(self, batch: list) -> dict:
    #     s, a, r, ns, d, t, dp, ndp, ge = batch
    #     return {
    #         'states': s,
    #         'actions': a,
    #         'rewards': r,
    #         'next_states': ns,
    #         'dones': d,
    #         'truncs': t,
    #         'state_decision_points': dp,
    #         'next_state_decision_points': ndp,
    #         'gamma_exps': ge
    #     }
