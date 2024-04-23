from tqdm import tqdm
import torch
import torch.nn as nn
from corerl.component.buffer.factory import init_buffer
from corerl.component.critic.factory import init_q_critic, init_v_critic
from corerl.component.network.utils import ensemble_mse
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseGVF(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, train_transitions: list[tuple], test_transitions: list[tuple]):
        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma

        self.input_dim = len(train_transitions[0][0])
        self.action_dim = len(train_transitions[0][1])

        self.buffer.load(train_transitions)
        self.test_buffer.load(test_transitions)

        self.train_losses = []
        self.test_losses = []

        self.gvf = None

    @abstractmethod
    def compute_gvf_loss(self, batch: dict, with_grad: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)
        loss = self.compute_gvf_loss(batch, with_grad=True)
        self.gvf.update(loss)
        self.train_losses.append(loss.detach().numpy())

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


class SimpleGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, train_transitions: list[tuple], test_transitions: list[tuple]):
        super().__init__(cfg, train_transitions, test_transitions)
        self.gvf = init_v_critic(cfg.critic, self.input_dim)

    def compute_gvf_loss(self, batch: dict, with_grad: bool = False) -> list[torch.Tensor]:
        def _compute_gvf_loss():
            states, actions, rewards, next_states, dones = (batch['states'], batch['actions'],
                                                            batch['rewards'], batch['next_states'], batch['dones'])
            next_v = self.gvf.get_v_target(next_states)
            target = rewards + (1 - dones) * self.gamma * next_v
            _, v_ens = self.gvf.get_qs(states, actions, with_grad=True)
            loss = ensemble_mse(target, v_ens)
            return loss

        if with_grad:
            return _compute_gvf_loss()
        else:
            with torch.no_grad():
                return _compute_gvf_loss()
