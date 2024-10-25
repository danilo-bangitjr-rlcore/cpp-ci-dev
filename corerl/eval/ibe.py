"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import numpy as np
import torch
from omegaconf import DictConfig

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.eval.base_eval import BaseEval
from corerl.data.data import TransitionBatch
from corerl.component.network.utils import ensemble_mse, to_np


class IBE(BaseEval):
    def __init__(self, cfg: DictConfig, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")

        self.agent = kwargs['agent']
        state_dim = self.agent.state_dim
        action_dim = self.agent.action_dim
        self.n_updates = cfg.n_updates
        self.gamma = cfg.gamma
        self.model = init_custom_network(cfg.network, state_dim + action_dim, output_dim=1)
        self.optimizer = init_optimizer(
            cfg.optimizer,
            list(self.model.parameters(independent=True)),
            ensemble=True
        )

        self.losses = []
        self.bes = []  # the bellman errors
        self.bes_changes = []  # the change of bellman errors
        self.bes_changes_smoothed = []  # the change of bellman errors

    def _get_delta(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.n_step_reward
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        gamma_exp_batch = batch.gamma_exponent
        dp_mask = batch.boot_state_dp

        next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
        # For the 'Anytime' paradigm, only states at decision points can sample next_actions
        # If a state isn't at a decision point, its next_action is set to the current action
        with torch.no_grad():
            next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

        if self.agent.ensemble_targets:
            _, next_q = self.agent.q_critic.get_qs_target(
                [next_state_batch], [next_actions],
            )
        else:
            next_q = self.agent.q_critic.get_q_target([next_state_batch], [next_actions])

        # N-Step SARSA update with variable 'N', thus 'reward_batch' is an n_step reward
        # and the exponent on gamma, 'gamma_exp_batch', depends on 'n'
        target = reward_batch + mask_batch * (self.gamma ** gamma_exp_batch) * next_q
        _, q_ens = self.agent.q_critic.get_qs([state_batch], [action_batch], with_grad=True)

        delta = target - q_ens

        return delta

    def get_loss(self, batch: TransitionBatch) -> torch.Tensor:
        delta = self._get_delta(batch)
        state_batch = batch.state
        action_batch = batch.action
        sa = torch.concatenate((state_batch, action_batch), dim=1)
        predictions = self.model(sa)

        loss = ensemble_mse(delta, predictions)
        return loss

    def do_eval(self, **kwargs) -> None:
        # train the model
        for _ in range(self.n_updates):
            batches = self.agent.critic_buffer.sample()
            batch = batches[0]
            loss = self.get_loss(batch)

            self.optimizer.zero_grad()
            for i in range(len(loss)):
                loss[i].backward(inputs=list(self.model.parameters(independent=True)[i]))

            self.optimizer.step()
            loss = [l.detach().item() for l in loss]
            self.losses.append(loss)

        # estimate the bellman error on a batch
        batches = self.agent.critic_buffer.sample()
        batch = batches[0]
        be = self._estimate_be(batch)
        self.bes.append(be.mean())
        if len(self.bes) > 1:
            self.bes_changes.append(self.bes[-1] - self.bes[-2])
        else:
            self.bes_changes.append(self.bes[-1])

    def _estimate_be(self, batch: TransitionBatch) -> np.ndarray:
        state_batch = batch.state
        action_batch = batch.action

        sa = torch.concatenate((state_batch, action_batch), dim=1)
        predictions = self.model(sa)
        delta = self._get_delta(batch)

        be_batch = 2 * predictions * delta - torch.square(predictions)
        mean_be = torch.squeeze(torch.mean(be_batch, axis=1))

        return to_np(mean_be).astype(np.float64)

    def get_stats(self) -> dict:
        smoothed_change = []
        ary_bes_changes = np.array(self.bes_changes, dtype=np.float64)
        for i in range(0, min(9, len(ary_bes_changes))):
            smoothed_change.append(ary_bes_changes[:i+1].mean())
        for i in range(10, len(ary_bes_changes)):
            smoothed_change.append(ary_bes_changes[i-9:i+1].mean())
        stats = {
            'bellman_error': self.bes,
            'ibe_change': self.bes_changes,
            'ibe_smoothed_change': smoothed_change
        }
        return stats
