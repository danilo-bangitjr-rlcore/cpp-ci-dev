from tqdm import tqdm
import torch
import torch.nn as nn
from corerl.component.buffer.factory import init_buffer
from corerl.component.critic.factory import init_q_critic, init_v_critic
from corerl.component.network.utils import ensemble_mse
from corerl.data.data import TransitionBatch, Transition
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseGVF(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.gamma = cfg.gamma

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.endo_obs_col_names = cfg.endo_obs_col_names
        self.endo_inds = list(range(len(self.endo_obs_col_names)))
        self.num_gvfs = len(self.endo_inds)

        self.train_losses = []
        self.test_losses = []

        self.gvf = None

    def update_train_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def update_test_buffer(self, transition: Transition) -> None:
        self.test_buffer.feed(transition)

    @abstractmethod
    def compute_gvf_loss(self, batch: dict, with_grad: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def update(self):
        batch = self.buffer.sample()
        loss = self.compute_gvf_loss(batch, with_grad=True)
        self.gvf.update(loss)
        #self.train_losses.append(loss.detach().numpy())

    def train(self):
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update()
            self.get_test_loss()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        return self.train_losses, self.test_losses

    def get_test_loss(self):
        batch = self.test_buffer.sample_batch()
        loss = self.compute_gvf_loss(batch)
        self.test_losses.append(loss.detach().numpy())

    def get_num_gvfs(self):
        return self.num_gvfs


class SimpleGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        super().__init__(cfg, input_dim, action_dim, **kwargs)
        self.gvf = init_v_critic(cfg.critic, self.input_dim, self.num_gvfs)

    def compute_gvf_loss(self, batch: TransitionBatch, with_grad: bool = False) -> list[torch.Tensor]:
        def _compute_gvf_loss():
            state_batch = batch.state
            action_batch = batch.action
            cumulant_batch = batch.reward
            next_state_batch = batch.boot_state
            mask_batch = 1 - batch.terminated
            gamma_exp_batch = batch.gamma_exponent

            next_v = self.gvf.get_v_target(next_state_batch)
            target = cumulant_batch + (mask_batch * (self.gamma ** gamma_exp_batch) * next_v)
            _, v_ens = self.gvf.get_vs(state_batch, action_batch, with_grad=True)
            loss = ensemble_mse(target, v_ens)
            return loss

        if with_grad:
            return _compute_gvf_loss()
        else:
            with torch.no_grad():
                return _compute_gvf_loss()


class QGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        
        super().__init__(cfg, input_dim, action_dim, **kwargs)
        self.gvf = init_q_critic(cfg.critic, self.input_dim, self.action_dim, self.num_gvfs)
        self.agent = kwargs["agent"]

    def compute_gvf_loss(self, batch: TransitionBatch, with_grad: bool = False) -> list[torch.Tensor]:
        def _compute_gvf_loss():
            state_batch = batch.state
            action_batch = batch.action
            cumulant_batch = batch.reward
            next_state_batch = batch.boot_state
            mask_batch = 1 - batch.terminated
            gamma_exp_batch = batch.gamma_exponent
            dp_mask = batch.boot_decision_point

            next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)
            next_q = self.gvf.get_q_target(next_state_batch, next_actions)
            target = cumulant_batch + (mask_batch * (self.gamma ** gamma_exp_batch) * next_q)
            _, q_ens = self.gvf.get_qs(state_batch, action_batch, with_grad=True)
            loss = ensemble_mse(target, q_ens)
            return loss

        if with_grad:
            return _compute_gvf_loss()
        else:
            with torch.no_grad():
                return _compute_gvf_loss()


