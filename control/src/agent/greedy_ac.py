import torch
import os
from src.agent.base import BaseAC
from src.network.factory import init_policy_network, init_optimizer


class GreedyAC(BaseAC):
    def __init__(self, cfg):
        super(GreedyAC, self).__init__(cfg)
        self.tau = self.cfg.tau
        self.rho = self.cfg.rho # percentage of action used for update
        self.num_samples = self.cfg.n

        # use the same network as in actor
        self.sampler = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim, cfg.action_scale, cfg.action_bias, cfg.activation, cfg.layer_init)
        self.sampler_optim = init_optimizer(cfg.optimizer, list(self.sampler.parameters()), cfg.lr_actor)

        self.gac_a_dim = self.action_dim

    def inner_update(self):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], data['obs2'], 1 - data['done']
        
        # critic update
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)
        target = reward_batch + mask_batch * self.gamma * next_q
        q_value, _ = self.get_q_value(state_batch, action_batch, with_grad=True)
        q_loss = torch.nn.functional.mse_loss(target, q_value)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # actor update
        sample_actions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                a, _, _ = self.sampler(state_batch)
                sample_actions.append(a)
        # sample_actions = torch.cat(sample_actions, dim=1).reshape((-1, self.action_dim))
        sample_actions = torch.cat(sample_actions, dim=1).reshape((-1, self.gac_a_dim))
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py
        q_values, _ = self.get_q_value(repeated_states, sample_actions, with_grad=False)
        q_values = q_values.reshape(self.batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        best_ind = sorted_q[:, :int(self.rho * self.num_samples)]
        # best_ind = best_ind.repeat_interleave(self.action_dim, -1)
        best_ind = best_ind.repeat_interleave(self.gac_a_dim, -1)

        # sample_actions = sample_actions.reshape(self.batch_size, self.num_samples, self.action_dim)
        sample_actions = sample_actions.reshape(self.batch_size, self.num_samples, self.gac_a_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        samples = int(self.rho * self.num_samples)
        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        # best_actions = torch.reshape(best_actions, (-1, self.action_dim))
        best_actions = torch.reshape(best_actions, (-1, self.gac_a_dim))

        logp, _ = self.actor.log_prob(stacked_s_batch, best_actions)
        pi_loss = -logp.mean()
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

        # sampler entropy update
        stacked_s_batch = state_batch.repeat_interleave(self.num_samples, dim=0)
        stacked_s_batch = stacked_s_batch.reshape(-1, self.state_dim)
        # sample_actions = sample_actions.reshape(-1, self.action_dim)
        sample_actions = sample_actions.reshape(-1, self.gac_a_dim)

        sampler_entropy, _ = self.sampler.log_prob(stacked_s_batch, sample_actions)
        with torch.no_grad():
            sampler_entropy *= sampler_entropy
        sampler_entropy = sampler_entropy.reshape(self.batch_size, self.num_samples, 1)
        sampler_entropy = -sampler_entropy[:, 0, :]
        sampler_entropy = -1 * sampler_entropy

        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        logp, _ = self.sampler.log_prob(stacked_s_batch, best_actions)
        sampler_loss = logp.reshape(self.batch_size, samples, 1)
        sampler_loss = -1 * (sampler_loss.mean(axis=1) + self.tau * sampler_entropy).mean()

        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        self.sampler_optim.step()

    def save(self):
        super(GreedyAC, self).save()
        parameters_dir = self.parameters_dir
        path = os.path.join(parameters_dir, "sampler_net")
        torch.save(self.sampler.state_dict(), path)
    
        path = os.path.join(parameters_dir, "sampler_opt")
        torch.save(self.sampler_optim.state_dict(), path)


class GreedyACDiscrete(GreedyAC):
    def __init__(self, cfg):
        super(GreedyACDiscrete, self).__init__(cfg)
        self.gac_a_dim = 1
        