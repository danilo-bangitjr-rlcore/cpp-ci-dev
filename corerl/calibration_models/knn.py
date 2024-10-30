import torch
import logging
import numpy as np
import sklearn.neighbors as skn

from tqdm import tqdm
from omegaconf import DictConfig

from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import to_np
from corerl.calibration_models.base import BaseCalibrationModel
from corerl.data.data import Transition, TransitionBatch

log = logging.getLogger(__name__)


class KNNCalibrationModel(BaseCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info: dict):
        super().__init__(cfg, train_info)
        self.train_transitions = train_info['train_transitions_cm']
        test_transitions = train_info['test_transitions_cm']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(self.train_transitions)
        self.test_buffer.load(test_transitions)

        self.state_dim = len(self.train_transitions[0].state)
        self.action_dim = len(self.train_transitions[0].action)
        self.endo_dim = len(self.train_transitions[0].obs[self.endo_inds])
        self.output_dim = cfg.output_dim

        self.learn_metric = cfg.learn_metric
        self.include_actions = cfg.include_actions  # whether to include actions in the learned representation
        self.metric = None
        self.beta = cfg.beta
        self.zeta = cfg.zeta
        self.num_neighbors = cfg.num_neighbors
        self.train_itr = cfg.train_itr

        self.model, self.optimizer = self._init_metric(cfg)


    def _init_metric(self, cfg: DictConfig):
        if self.learn_metric:  # learn a laplacian
            log.info("Learning Laplacian representation...")
            if self.include_actions:
                input_dim = self.state_dim + self.action_dim
            else:
                input_dim = self.state_dim

            model = init_custom_network(cfg.model, input_dim=input_dim, output_dim=self.output_dim)
            optimizer = init_optimizer(cfg.optimizer, list(model.parameters()))

            return model, optimizer

        raise NotImplementedError

    def _get_rep(self, state: np.ndarray) -> np.ndarray:
        if self.learn_metric:
            state_tensor = torch.from_numpy(state)
            with torch.no_grad():
                rep_tensor = self.model(state_tensor)
            rep = to_np(rep_tensor)
            return rep
        else:
            return state

    def train(self, loss_avg=100) -> None:
        if self.learn_metric:
            log.info("Learning Laplacian representation...")
            losses = []
            pbar = tqdm(range(self.train_itr))
            for _ in pbar:
                batch = self.buffer.sample_mini_batch(self.batch_size)[0]
                loss = self._laplacian_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()

                loss = loss.detach().item()
                losses.append(loss)
                if len(losses) > loss_avg:
                    avg_loss = sum(losses[-loss_avg:]) / loss_avg
                    pbar.set_description(f"Avg loss over last {loss_avg} iterations:: {avg_loss:.4f}")

                self.optimizer.step(closure=lambda: 0.)

        log.info("Constructing lookup...")
        self._construct_lookup_continuous(self.train_transitions)

    def _laplacian_loss(self, batch: TransitionBatch) -> torch.Tensor:
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

    def _construct_lookup_discrete(self, transitions: list[Transition]) -> None:
        """
        This function would be used to cache nearest neighbours in discrete action settings
        """
        raise NotImplementedError

    def _construct_lookup_continuous(self, transitions: list[Transition]) -> None:
        state_reps = [self._get_rep(t.state) for t in transitions]
        state_reps = np.array(state_reps)
        actions = [t.action for t in transitions]
        actions = np.array(actions)
        reps = np.concatenate((state_reps, actions), axis=1)
        self.tree = skn.KDTree(reps)

    def _get_next_endo_obs(self, state: np.ndarray, action: np.ndarray, kwargs: dict) -> np.ndarray:
        state_rep = self._get_rep(state)
        rep = np.concatenate((state_rep, action))
        distances, indices = self.tree.query([rep], k=self.num_neighbors)
        distances = np.squeeze(distances)
        indices = np.squeeze(indices)
        probs = 1 - (distances / np.sum(distances))
        probs = probs / np.sum(probs)
        sampled_idx = np.random.choice(indices, p=probs)
        sampled_transition = self.train_transitions[sampled_idx]
        next_obs = sampled_transition.next_obs[self.endo_inds]
        next_obs = np.expand_dims(next_obs, 0)

        return next_obs
