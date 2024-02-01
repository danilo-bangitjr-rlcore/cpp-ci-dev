import torch
import os
from src.agent.base import BaseAC
from src.network.factory import init_policy_network, init_optimizer


class GreedyAC(BaseAC):
    def __init__(self, cfg, average_entropy=True):
        super(GreedyAC, self).__init__(cfg)
        self.tau = self.cfg.tau
        self.rho = self.cfg.rho # percentage of action used for update
        self.rho_proposal = self.rho * self.cfg.prop_rho_mult # percentage of action used for update proposal
        self.num_samples = self.cfg.n
        self.average_entropy = average_entropy

        # use the same network as in actor
        self.sampler = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                           cfg.beta_parameter_bias, cfg.activation,
                                           cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
        self.sampler_optim = init_optimizer(cfg.optimizer, list(self.sampler.parameters()), cfg.lr_actor)

        self.gac_a_dim = self.action_dim
        self.top_action = int(self.rho * self.num_samples)
        self.top_action_proposal = int(self.rho_proposal * self.num_samples)

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']
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
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        with torch.no_grad():
            sample_actions, _, _ = self.sampler(repeated_states)

        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py
        q_values, _ = self.get_q_value(repeated_states, sample_actions, with_grad=False)
        q_values = q_values.reshape(self.batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        best_ind = sorted_q[:, :self.top_action]
        best_ind = best_ind.repeat_interleave(self.gac_a_dim, -1)

        sample_actions = sample_actions.reshape(self.batch_size, self.num_samples, self.gac_a_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_action, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.gac_a_dim))

        logp, _ = self.actor.log_prob(stacked_s_batch, best_actions)
        pi_loss = -logp.mean()
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

        # proposal policy update
        if self.tau != 0:
            # sample_actions = sample_actions.reshape(-1, self.action_dim)
            sample_actions = sample_actions.reshape(-1, self.gac_a_dim)

            sampler_entropy, _ = self.sampler.log_prob(repeated_states, sample_actions)
            with torch.no_grad():
                sampler_entropy *= sampler_entropy
            sampler_entropy = sampler_entropy.reshape(self.batch_size, self.num_samples, 1)

            if self.average_entropy:
                sampler_entropy = -sampler_entropy.mean(axis=1)
            else:
                sampler_entropy = -sampler_entropy[:, 0, :]

            logp, _ = self.sampler.log_prob(stacked_s_batch, best_actions)
            sampler_loss = logp.reshape(self.batch_size, self.top_action, 1)
            sampler_loss = -1 * (sampler_loss.mean(axis=1) + self.tau * sampler_entropy).mean()
        else:
            best_ind_proposal = sorted_q[:, :self.top_action_proposal]
            best_ind_proposal = best_ind_proposal.repeat_interleave(self.gac_a_dim, -1)
            best_actions_proposal = torch.gather(sample_actions, 1, best_ind_proposal)
            stacked_s_batch_proposal = state_batch.repeat_interleave(self.top_action_proposal, dim=0)
            best_actions_proposal = torch.reshape(best_actions_proposal, (-1, self.gac_a_dim))

            logp, _ = self.sampler.log_prob(stacked_s_batch_proposal, best_actions_proposal)
            sampler_loss = logp.reshape(self.batch_size, self.top_action_proposal, 1)
            sampler_loss = -1 * (sampler_loss.mean(axis=1)).mean()


        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        self.sampler_optim.step()
        return data

    def save(self):
        super(GreedyAC, self).save()
        parameters_dir = self.parameters_dir
        path = os.path.join(parameters_dir, "sampler_net")
        torch.save(self.sampler.state_dict(), path)
    
        path = os.path.join(parameters_dir, "sampler_opt")
        torch.save(self.sampler_optim.state_dict(), path)

    def agent_debug_info(self, observation_tensor, action_tensor, pi_info, env_info):
        i_log = super(GreedyAC, self).agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
        _, _, prop_pi_info = self.sampler(observation_tensor, self.cfg.debug)
        i_log["proposal_info"] = prop_pi_info
        return i_log


class GreedyACDiscrete(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(GreedyACDiscrete, self).__init__(cfg, average_entropy)
        self.gac_a_dim = 1
        self.top_action = 1