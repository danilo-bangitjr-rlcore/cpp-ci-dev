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

        # use the same actor_network as in actor
        self.sampler = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                           cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                           cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
        self.sampler_optim = init_optimizer(cfg.optimizer, list(self.sampler.parameters()), cfg.lr_actor)
        self.gac_a_dim = self.action_dim
        self.top_action = int(self.rho * self.num_samples)
        self.top_action_proposal = int(self.rho_proposal * self.num_samples)

        if self.discrete_control:
            self.top_action = 1
            if self.num_samples > self.gac_a_dim:
                self.get_policy_update_data = self.get_policy_update_data_discrete
                self.num_samples = self.gac_a_dim
                self.top_action_proposal = self.gac_a_dim


    def critic_loss(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)
        target = reward_batch + mask_batch * self.gamma * next_q
        temp, q_ens = self.get_q_value(state_batch, action_batch, with_grad=True)
        # for i in range(len(temp)):
        #     print(temp[i], [q_ens[j][i] for j in range(len(q_ens))])
        q_loss = self.ensemble_mse(target, q_ens) #torch.nn.functional.mse_loss(target, q_value)
        return q_loss, next_action

    def sort_q_value(self, repeated_states, sample_actions, batch_size):
        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py
        q_values, _ = self.get_q_value(repeated_states, sample_actions, with_grad=False)
        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q

    def get_policy_update_data(self, state_batch):
        batch_size = state_batch.shape[0]
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)
        with torch.no_grad():
            sample_actions, _, _ = self.sampler(repeated_states)

        sorted_q = self.sort_q_value(repeated_states, sample_actions, batch_size)
        best_ind = sorted_q[:, :self.top_action]
        best_ind = best_ind.repeat_interleave(self.gac_a_dim, -1)

        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.gac_a_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_action, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.gac_a_dim))
        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions

    def get_policy_update_data_discrete(self, state_batch):
        batch_size = state_batch.shape[0]
        repeated_states = state_batch.repeat_interleave(self.gac_a_dim, dim=0)
        actions = torch.arange(self.gac_a_dim).reshape((1, -1))
        actions = actions.repeat_interleave(batch_size, dim=0).reshape((-1, 1))
        a_onehot = torch.FloatTensor(actions.size()[0], self.gac_a_dim)
        a_onehot.zero_()
        sample_actions = a_onehot.scatter_(1, actions, 1)

        sorted_q = self.sort_q_value(repeated_states, sample_actions, batch_size)
        best_ind = sorted_q[:, :self.top_action]
        best_ind = best_ind.repeat_interleave(self.gac_a_dim, -1)

        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.gac_a_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_action, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.gac_a_dim))
        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions

    def actor_loss(self, state_batch):
        repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions = self.get_policy_update_data(state_batch)
        logp, _ = self.actor.log_prob(stacked_s_batch, best_actions)
        pi_loss = -logp.mean()
        return pi_loss, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, logp

    def proposal_loss(self, sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch):
        batch_size = state_batch.shape[0]
        # proposal policy update
        if self.tau != 0:
            # sample_actions = sample_actions.reshape(-1, self.action_dim)
            sample_actions = sample_actions.reshape(-1, self.gac_a_dim)

            sampler_entropy, _ = self.sampler.log_prob(repeated_states, sample_actions)
            with torch.no_grad():
                sampler_entropy *= sampler_entropy
            sampler_entropy = sampler_entropy.reshape(batch_size, self.num_samples, 1)

            if self.average_entropy:
                sampler_entropy = -sampler_entropy.mean(axis=1)
            else:
                sampler_entropy = -sampler_entropy[:, 0, :]

            logp, _ = self.sampler.log_prob(stacked_s_batch, best_actions)
            sampler_loss = logp.reshape(batch_size, self.top_action, 1)
            sampler_loss = -1 * (sampler_loss.mean(axis=1) + self.tau * sampler_entropy).mean()
        else:
            best_ind_proposal = sorted_q[:, :self.top_action_proposal]
            best_ind_proposal = best_ind_proposal.repeat_interleave(self.gac_a_dim, -1)
            best_actions_proposal = torch.gather(sample_actions, 1, best_ind_proposal)
            stacked_s_batch_proposal = state_batch.repeat_interleave(self.top_action_proposal, dim=0)
            best_actions_proposal = torch.reshape(best_actions_proposal, (-1, self.gac_a_dim))

            logp, _ = self.sampler.log_prob(stacked_s_batch_proposal, best_actions_proposal)
            sampler_loss = logp.reshape(batch_size, self.top_action_proposal, 1)
            sampler_loss = -1 * (sampler_loss.mean(axis=1)).mean()
        return sampler_loss

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']
        # critic update
        q_loss, _ = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.critic_optimizer.zero_grad()
        self.ensemble_critic_loss_backward(q_loss) #q_loss.backward()
        self.critic_optimizer.step()

        # actor update
        pi_loss, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, _ = self.actor_loss(state_batch)
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

        sampler_loss = self.proposal_loss(sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch)
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