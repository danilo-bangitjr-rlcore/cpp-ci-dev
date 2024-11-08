import logging
import os
import torch
import torch.nn as nn
import numpy as np

from typing import List
from tqdm import tqdm
from omegaconf import DictConfig
from pathlib import Path

from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.base import BaseCalibrationModel
from corerl.utils.device import device


log = logging.getLogger(__name__)


class ShortHorizonDelta(BaseCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info: dict):
        super().__init__(cfg, train_info)
        train_transitions = train_info['train_transitions_cm']
        self.rng = np.random.RandomState(cfg.seed)

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)

        input_dim = len(train_transitions[0].state)
        action_dim = len(train_transitions[0].action)
        output_dim = len(train_transitions[0].obs[self.endo_inds])

        self.model = init_custom_network(cfg.model, input_dim=input_dim + action_dim, output_dim=output_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

    def _update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)[0]
        obs_batch, state_batch, action_batch, next_obs_batch = batch.obs, batch.state, batch.action, batch.next_obs
        # we only predict the next endogenous component of the observation
        endo_next_obs_batch = next_obs_batch[:, self.endo_inds]
        delta_endo_next_obs_batch = endo_next_obs_batch - obs_batch[:, self.endo_inds]
        x = torch.concat((state_batch, action_batch), dim=1)
        prediction = self.model(x)
        loss = nn.functional.mse_loss(prediction, delta_endo_next_obs_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(closure=lambda: 0.)

        self.train_losses.append(loss.detach().numpy())

    def train(self) -> List[float]:
        log.info('Training model...')
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self._update()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))
        return self.train_losses

    def get_prediction(
            self,
            obs: torch.Tensor,
            state: torch.Tensor,
            action: torch.Tensor,
            with_grad: bool = False
    ):
        x = torch.concat((state, action), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        y += obs[:, self.endo_inds]
        y = torch.clamp(y, min=0, max=1)
        return y

    def _get_next_endo_obs(self, state, action, kwargs):
        state_tensor = tensor(state).reshape((1, -1))
        action_tensor = tensor(action).reshape((1, -1))
        return self.get_prediction(kwargs["prev_obs"], state_tensor, action_tensor)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        net_path = os.path.join(path, "model")
        torch.save(self.model.state_dict(), net_path)
        return

    def load(self, path: Path) -> None:
        net_path = os.path.join(path, "model")
        self.model.load_state_dict(torch.load(net_path, map_location=device.device))
        return
