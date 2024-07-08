from tqdm import tqdm
import torch
import torch.nn as nn
from corerl.component.buffer.factory import init_buffer
from corerl.component.critic.factory import init_q_critic, init_v_critic
from corerl.component.network.utils import ensemble_mse
from corerl.data.data import TransitionBatch, Transition
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Optional


class BaseGVF(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.gamma = cfg.gamma

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.endo_obs_names = cfg.endo_obs_names
        self.endo_inds = cfg.endo_inds
        assert len(self.endo_obs_names) > 0, "In config/env/<env_name>.yaml, define 'endo_obs_names' to be a list of the names of the endogenous variables in the observation"
        assert len(self.endo_inds) > 0, "In config/env/<env_name>.yaml, define 'endo_inds' to be a list of the indices of the endogenous variables within the environment's observation vector"
        assert len(self.endo_obs_names) == len(self.endo_inds), "The length of self.endo_obs_names and self.endo_inds should be the same and the ordering of the indices should correspond to the ordering of the variable names"
        self.num_gvfs = len(self.endo_inds)

        self.train_losses = []
        self.test_losses = []

        self.gvf = None

    def update_train_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def update_test_buffer(self, transition: Transition) -> None:
        self.test_buffer.feed(transition)

    @abstractmethod
    def compute_gvf_loss(self, batch: dict, cumulant_inds: Optional[list[int]] = None, with_grad: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def update(self, cumulant_inds: Optional[list[int]] = None):
        batch = self.buffer.sample()
        loss = self.compute_gvf_loss(batch, cumulant_inds=cumulant_inds, with_grad=True)
        self.gvf.update(loss)
        #self.train_losses.append(loss.detach().numpy())

    def train(self, cumulant_inds: Optional[list[int]] = None):
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update(cumulant_inds=cumulant_inds)
            self.get_test_loss()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        return self.train_losses, self.test_losses

    def get_test_loss(self, cumulant_inds: Optional[list[int]] = None):
        batch = self.test_buffer.sample_batch()
        loss = self.compute_gvf_loss(batch, cumulant_inds=cumulant_inds)
        self.test_losses.append(loss.detach().numpy())

    def get_num_gvfs(self):
        return self.num_gvfs


class SimpleGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        super().__init__(cfg, input_dim, action_dim, **kwargs)
        self.gvf = init_v_critic(cfg.critic, self.input_dim, self.num_gvfs)

    def compute_gvf_loss(self, batch: TransitionBatch, cumulant_inds: Optional[list[int]] = None, with_grad: bool = False) -> list[torch.Tensor]:
        def _compute_gvf_loss(cumulant_inds: Optional[list[int]] = None):
            state_batch = batch.state
            action_batch = batch.action
            cumulant_batch = batch.n_step_reward
            next_state_batch = batch.boot_state
            mask_batch = 1 - batch.terminated
            gamma_exp_batch = batch.gamma_exponent
            cumulant_batch = batch.n_step_cumulants
            if cumulant_inds:
                cumulant_batch = cumulant_batch[:, cumulant_inds]

            next_v = self.gvf.get_v_target(next_state_batch)
            target = cumulant_batch + (mask_batch * (self.gamma ** gamma_exp_batch) * next_v)
            _, v_ens = self.gvf.get_vs(state_batch, action_batch, with_grad=True)
            loss = ensemble_mse(target, v_ens)
            return loss

        if with_grad:
            return _compute_gvf_loss(cumulant_inds=cumulant_inds)
        else:
            with torch.no_grad():
                return _compute_gvf_loss(cumulant_inds=cumulant_inds)


class QGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        
        super().__init__(cfg, input_dim, action_dim, **kwargs)
        self.gvf = init_q_critic(cfg.critic, self.input_dim, self.action_dim, self.num_gvfs)
        self.agent = kwargs["agent"]

    def compute_gvf_loss(self, batch: TransitionBatch, cumulant_inds: Optional[list[int]] = None, with_grad: bool = False) -> list[torch.Tensor]:
        def _compute_gvf_loss(cumulant_inds: Optional[list[int]] = None):
            state_batch = batch.state
            action_batch = batch.action
            next_state_batch = batch.boot_state
            mask_batch = 1 - batch.terminated
            gamma_exp_batch = batch.gamma_exponent
            dp_mask = batch.boot_state_dp
            cumulant_batch = batch.n_step_cumulants
            if cumulant_inds:
                cumulant_batch = cumulant_batch[:, cumulant_inds]

            next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)
            next_q = self.gvf.get_q_target(next_state_batch, next_actions)
            target = cumulant_batch + (mask_batch * (self.gamma ** gamma_exp_batch) * next_q)
            _, q_ens = self.gvf.get_qs(state_batch, action_batch, with_grad=True)
            loss = ensemble_mse(target, q_ens)
            return loss

        if with_grad:
            return _compute_gvf_loss(cumulant_inds=cumulant_inds)
        else:
            with torch.no_grad():
                return _compute_gvf_loss(cumulant_inds=cumulant_inds)


