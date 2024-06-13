import torch

from abc import ABC, abstractmethod
from omegaconf import DictConfig

from corerl.data import Trajectory
from corerl.agent.base import BaseAgent


class BaseCalibrationModel(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, train_info: dict):
        raise NotImplementedError

    @abstractmethod
    def do_test_rollout(self, traj: Trajectory, start_idx):
        raise NotImplementedError

    @abstractmethod
    def do_test_rollouts(self):
        raise NotImplementedError

    @abstractmethod
    def do_agent_rollout(self, traj: Trajectory, agent: BaseAgent):
        raise NotImplementedError

    @abstractmethod
    def do_agent_rollouts(self, agent):
        raise NotImplementedError


class NNCalibrationModel(BaseCalibrationModel):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, with_grad: bool = False):
        raise NotImplementedError
