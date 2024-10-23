import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig

from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.data.data import Transition

log = logging.getLogger(__name__)


class OneStepModel:
    def __init__(self, cfg: DictConfig, train_transitions: list[Transition], test_transitions: list[Transition]):
        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)
        self.test_buffer.load(test_transitions)

        input_dim = len(train_transitions[0].state)
        action_dim = len(train_transitions[0].action)
        output_dim = len(train_transitions[0].state)

        self.model = init_custom_network(cfg.model, input_dim=input_dim+action_dim, output_dim=output_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

    def get_prediction(self, state, action, with_grad=False):
        x = torch.concat((state, action), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def update(self):
        batches = self.buffer.sample_mini_batch(self.batch_size)
        batch = batches[0]

        prediction = self.get_prediction(batch.state, batch.action, with_grad=True)
        loss = nn.functional.mse_loss(prediction, batch.next_state)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(closure=lambda: 0.)

        self.train_losses.append(loss.detach().numpy())

    def train(self):
        log.info('Training model...')
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update()
            self.get_test_loss()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        return self.train_losses, self.test_losses

    def get_test_loss(self):
        batches = self.test_buffer.sample_batch()
        batch = batches[0]
        prediction = self.get_prediction(batch.state, batch.action, with_grad=False)
        loss = nn.functional.mse_loss(prediction, batch.next_state)
        self.test_losses.append(loss.detach().numpy())
