import torch
import torch.nn as nn
import numpy as np
import random
import sklearn.neighbors as skn

from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from typing import Optional

from corerl.agent.base import BaseAgent
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.base import BaseCalibrationModel
from corerl.data.data import Trajectory, Transition, TransitionBatch
from corerl.state_constructor.base import BaseStateConstructor
import corerl.calibration_models.utils as utils

import matplotlib.pyplot as plt


class KNNCalibrationModel(BaseCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info):
        self.test_trajectories = train_info['test_trajectories_cm']
        self.train_transitions = train_info['train_transitions_cm']
        test_transitions = train_info['test_transitions_cm']
        self.reward_func = train_info['reward_func']
        self.normalizer = train_info['normalizer']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(self.train_transitions)
        self.test_buffer.load(test_transitions)

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        self.state_dim = len(self.train_transitions[0].state)
        self.action_dim = len(self.train_transitions[0].action)
        self.endo_dim = len(self.train_transitions[0].obs[self.endo_inds])
        self.output_dim = cfg.output_dim

        self.learn_metric = cfg.learn_metric
        self.include_actions = cfg.include_actions  # whether to include actions in the learned representation
        self.metric = None
        self.model = None
        self.beta = cfg.beta
        self.zeta = cfg.zeta
        self.num_neighbors = cfg.num_neighbors
        self.train_itr = cfg.train_itr

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.num_test_rollouts = cfg.num_test_rollouts

        self._init_metric(cfg)

    # def _norm(self, a: np.ndarray | torch.Tensor) -> torch:
    #     if isinstance(a, torch.Tensor):
    #         return torch.norm(a, p=2, dim=1)
    #     elif isinstance(a, np.ndarray):
    #         return np.linalg.norm(a)
    #     else:
    #         raise TypeError('Must be a Tensor or numpy array')
    #
    # def _euclidean_distance(self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    #     if isinstance(a, torch.Tensor):
    #         assert isinstance(b, torch.Tensor)
    #     elif isinstance(a, np.ndarray):
    #         pass
    #     return np.linalg.norm(a - b)
    #
    # def _model_metric(self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    #     return self._euclidean_distance(self.model(a), self.model(b))

    def _init_metric(self, cfg) -> None:
        if self.learn_metric:  # learn a laplacian
            print("Learning Laplacian representation...")
            if self.include_actions:
                input_dim = self.state_dim + self.action_dim
            else:
                input_dim = self.state_dim

            self.model = init_custom_network(cfg.model, input_dim=input_dim, output_dim=self.output_dim)
            self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))
        #     self.metric = self._model_metric
        # else:
        #     self.metric = self._euclidean_distance

    def _get_rep(self, state: np.ndarray) -> np.ndarray:
        if self.learn_metric:
            state_tensor = torch.from_numpy(state)
            with torch.no_grad():

                rep_tensor = self.model(state_tensor)

            rep = to_np(rep_tensor)
            return rep
        else:
            return state

    def train(self):
        if self.learn_metric:
            print("Learning Laplacian representation...")
            losses = []
            pbar = tqdm(range(self.train_itr))
            for _ in pbar:
                batch = self.buffer.sample(self.batch_size)
                loss = self._laplacian_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()

                loss = loss.detach().item()
                losses.append(loss)
                if len(losses) > 100:
                    avg_loss = sum(losses[-100:]) / 100
                    pbar.set_description(f"Loss: {avg_loss:.4f}")

                self.optimizer.step()

        print("Contructing lookup...")
        self._construct_lookup_continuous(self.train_transitions)

    def _laplacian_loss(self, batch: TransitionBatch):
        """
        TODO: have this include actions in representations. Will need to modify Transition class to save the next transition.
        """
        # leaving for when I'm fresh
        state_batch = batch.state
        next_state_batch = batch.next_state

        batch_size = state_batch.size(0)

        state_rep = self.model(state_batch)
        next_state_rep = self.model(next_state_batch)

        norm = torch.norm(state_rep - next_state_rep, p=2, dim=1)
        attractive_loss = torch.mean(torch.pow(norm, 2))

        repulsive_loss_ = torch.pow(torch.matmul(state_rep, torch.transpose(state_rep, 0, 1)), 2)

        state_norms = torch.pow(torch.norm(state_rep, p=2, dim=1), 2)
        state_norms = torch.unsqueeze(state_norms, 1)

        repeated_state_norms = state_norms.repeat(1, batch_size)

        repulsive_loss = torch.mean(self.beta * repulsive_loss_
                                    - self.zeta * repeated_state_norms
                                    - self.zeta * torch.transpose(repeated_state_norms, 0, 1))

        loss = attractive_loss + repulsive_loss

        return loss

    def _construct_lookup_discrete(self, transitions: list[Transition]):
        raise NotImplementedError

    def _construct_lookup_continuous(self, transitions: list[Transition]):
        if self.learn_metric:
            states_np = [t.state for t in transitions]
            states_np = np.array(states_np)
            state_tensor = torch.from_numpy(states_np)
            with torch.no_grad():
                reps = self.model(state_tensor)
            reps = to_np(reps)
        else:
            reps = [t.state for t in transitions]
            reps = np.array(reps)

        self.tree = skn.KDTree(reps)


    def do_test_rollouts(self):
        pass

    def do_test_rollout(self):
        pass
