from tqdm import tqdm
import torch
import torch.nn as nn
from corerl.component.network.utils import to_np

from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer


class OneStepModel:
    def __init__(self, cfg, train_transitions, test_transitions):
        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.feed(train_transitions)
        self.test_buffer.feed(test_transitions)

        input_dim = len(train_transitions[0][0])
        action_dim = len(train_transitions[0][1])
        self.model = init_custom_network(cfg.model, input_dim=input_dim+action_dim, output_dim=input_dim)
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
        batch = self.buffer.sample_mini_batch(self.batch_size)
        state_batch, action_batch, next_state_batch = batch['states'], batch['actions'], batch['next_states']
        prediction = self.get_prediction(state_batch, action_batch, with_grad=True)
        loss = nn.functional.mse_loss(prediction, next_state_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(to_np(loss))

    def train(self):
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update()
            self.get_test_loss()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        return self.train_losses, self.test_losses

    def get_test_loss(self):
        batch = self.test_buffer.sample_batch()
        state_batch, action_batch, next_state_batch = batch['states'], batch['actions'], batch[
            'next_states']
        prediction = self.get_prediction(state_batch, action_batch, with_grad=False)
        loss = nn.functional.mse_loss(prediction, next_state_batch)
        self.test_losses.append(loss)
