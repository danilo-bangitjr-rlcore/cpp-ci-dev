import numpy as np
import torch
import os
from src.agent.greedy_ac import GreedyAC
from src.network.factory import init_policy_network, init_custom_network, init_optimizer
import src.network.torch_utils as torch_utils
from src.component.buffer import Buffer
from src.component.normalizer import init_normalizer


class GACPredictSuccess(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(GACPredictSuccess, self).__init__(cfg, average_entropy=average_entropy)
        if cfg.discrete_control:
            self.gac_a_dim = 1
            self.top_action = 1
            self.predict_action_encoder = init_normalizer("OneHot", self.action_dim)
        else:
            self.predict_action_encoder = init_normalizer("Identity", None)

        self.predict_model = init_custom_network("Softmax", cfg.device, self.state_dim+self.action_dim, cfg.hidden_critic, 2,
                                                 cfg.activation, "None", cfg.layer_init)
        self.predict_model_optim = init_optimizer(cfg.optimizer, list(self.predict_model.parameters()), cfg.lr_critic)

        self.safe_action_model = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                                     cfg.action_scale, cfg.action_bias, cfg.activation, cfg.head_activation, cfg.layer_init)
        self.safe_action_model_optim = init_optimizer(cfg.optimizer, list(self.safe_action_model.parameters()), cfg.lr_actor)

        self.safe_action_buffer = Buffer(cfg.buffer_size, cfg.batch_size, cfg.seed)

        self.optimal_reward = 1
        self.relax = 0.005
        self.start_safty_check = 1000

    def inner_update(self, trunc=False):
        data = super(GACPredictSuccess, self).inner_update(trunc=trunc)
        if self.total_steps == self.start_safty_check:
            for _ in range(self.start_safty_check):
                self.checking_models_update(data)
        elif self.total_steps > self.start_safty_check:
            self.checking_models_update(data)
        #
        # self.checking_models_update(data)

    def checking_models_update(self, data):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']
        """ Update success prediction model """
        x = torch.concat((state_batch, self.predict_action_encoder(action_batch)), dim=1)
        _, logp, _ = self.predict_model(x)
        r_loss = -logp.mean()
        self.predict_model_optim.zero_grad()
        r_loss.backward()
        self.predict_model_optim.step()

        """ Update action model """
        samples = self.safe_action_buffer.sample()
        if samples is not None:
            states, actions = samples
            states = torch_utils.tensor(self.state_normalizer(states), self.device)
            actions = torch_utils.tensor(actions, self.device)
            logp, _ = self.safe_action_model.log_prob(states, actions)
            a_loss = -logp.mean()
            self.safe_action_model_optim.zero_grad()
            a_loss.backward()
            self.safe_action_model_optim.step()

    def step(self):
        observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
        action_tensor, _, pi_info = self.get_policy(observation_tensor,
                                                    with_grad=False, debug=self.cfg.debug)

        """predict the success for the given policy"""
        safe, _, info = self.safty_check(observation_tensor, action_tensor)
        if not safe:
            with torch.no_grad():
                action_tensor, _, _ = self.safe_action_model(observation_tensor)

        action = torch_utils.to_np(action_tensor)[0]
        next_observation, reward, terminated, trunc, env_info = self.env.step(action)
        reset, truncate = self.update_stats(reward, terminated, trunc)

        """When using the known optimal action, do not add data to buffer"""
        self.buffer.feed([self.observation, action, reward, next_observation, int(terminated), int(truncate)])
        if reward >= self.optimal_reward-self.relax:
            self.safe_action_buffer.feed([self.observation, action])

        i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
        self.info_log.append(i_log)

        if self.cfg.render:
            self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'])
        else:
            env_info.pop('interval_log', None)
            i_log['critic_info'].pop('Q-function', None)

        self.update(trunc)

        if reset:
            next_observation, info = self.env.reset()
        self.observation = next_observation

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return

    def safty_check(self, obs, act):
        x = torch.concat((obs, self.predict_action_encoder(act)), dim=1)
        with torch.no_grad():
            safe, logp, info = self.predict_model(x)
        if self.total_steps < self.start_safty_check:
            return 1.0, 1.0, info
        safe = torch_utils.to_np(safe.squeeze(-1))[0]
        logp = torch_utils.to_np(logp)[0]
        return safe, logp, info


class GACwHardMemory(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(GACwHardMemory, self).__init__(cfg, average_entropy=average_entropy)
        self.reward_model = init_custom_network(cfg.critic, cfg.device, self.state_dim+self.action_dim, cfg.hidden_critic, 1,
                                               cfg.activation, "None", cfg.layer_init)
        self.reward_model_optim = init_optimizer(cfg.optimizer, list(self.reward_model.parameters()), cfg.lr_critic)

        self.known_best_action = None
        self.known_best_reward = -np.inf
        self.optimal_reward = 1
        self.relax = 0.02
        self.start_safty_check = 1000

    def inner_update(self, trunc=False):
        data = super(GACwHardMemory, self).inner_update(trunc=trunc)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']

        """ Update reward model """
        x = torch.concat((state_batch, action_batch), dim=1)
        r_pred = self.reward_model(x)
        r_loss = torch.nn.functional.mse_loss(reward_batch, r_pred)
        self.reward_model_optim.zero_grad()
        r_loss.backward()
        self.reward_model_optim.step()

    def step(self):
        observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
        action_tensor, _, pi_info = self.get_policy(observation_tensor,
                                                    with_grad=False, debug=self.cfg.debug)

        """predict the reward for the given policy"""
        safe, pessimistic_reward = self.safty_check(observation_tensor, action_tensor)
        if not safe and self.known_best_action is not None:
            action_tensor = self.known_best_action

        action = torch_utils.to_np(action_tensor)[0]
        next_observation, reward, terminated, trunc, env_info = self.env.step(action)
        reset, truncate = self.update_stats(reward, terminated, trunc)

        """When using the known optimal action, update the corresponding reward"""
        if not safe:
            self.known_best_reward = reward
        """Update known optimal action and reward"""
        if self.known_best_action is None or reward > self.known_best_reward:
            self.known_best_action = action_tensor
            self.known_best_reward = reward
        """When using the known optimal action, use the pessimistic reward for save and use for update"""
        if not safe:
            print(pessimistic_reward)
            reward = pessimistic_reward

        self.buffer.feed([self.observation, action, reward, next_observation, int(terminated), int(truncate)])

        i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
        self.info_log.append(i_log)

        if self.cfg.render:
            self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'])
        else:
            env_info.pop('interval_log', None)
            i_log['critic_info'].pop('Q-function', None)

        self.update(trunc)

        if reset:
            next_observation, info = self.env.reset()
        self.observation = next_observation

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return

    def safty_check(self, obs, act):
        if self.total_steps < self.start_safty_check:
            return True, 1.0
        x = torch.concat((obs, act), dim=1)
        with torch.no_grad():
            pred_r = torch_utils.to_np(self.reward_model(x).squeeze(-1))[0]
        safe = pred_r >= (self.optimal_reward - self.relax)
        return safe, pred_r


class GACwMemory(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(GACwMemory, self).__init__(cfg)
        self.reward_model = init_custom_network(cfg.critic, cfg.device, self.state_dim+self.action_dim, cfg.hidden_critic, 1,
                                               cfg.activation, "None", cfg.layer_init)
        self.reward_model_optim = init_optimizer(cfg.optimizer, list(self.reward_model.parameters()), cfg.lr_critic)
        # self.current_best_action = None

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']

        """ Update reward model """
        x = torch.concat((state_batch, action_batch), dim=1)
        r_pred = self.reward_model(x)
        r_loss = torch.nn.functional.mse_loss(reward_batch, r_pred)
        self.reward_model_optim.zero_grad()
        r_loss.backward()
        self.reward_model_optim.step()

        # critic update
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)
        target = reward_batch + mask_batch * self.gamma * next_q
        q_value, _ = self.get_q_value(state_batch, action_batch, with_grad=True)

        # """Add penalty of reward prediction""" # Maybe it's not what we need, only need to learn a good threshold
        # with torch.no_grad():
        #     x = torch.concat((next_state_batch, next_action), dim=1)
        #     next_r_pred = self.reward_model(x)
        # target += next_r_pred

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
        sample_actions = torch.cat(sample_actions, dim=1).reshape((-1, self.gac_a_dim))
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

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

        # sampler entropy update
        stacked_s_batch = state_batch.repeat_interleave(self.num_samples, dim=0)
        stacked_s_batch = stacked_s_batch.reshape(-1, self.state_dim)
        # sample_actions = sample_actions.reshape(-1, self.action_dim)
        sample_actions = sample_actions.reshape(-1, self.gac_a_dim)

        sampler_entropy, _ = self.sampler.log_prob(stacked_s_batch, sample_actions)
        with torch.no_grad():
            sampler_entropy *= sampler_entropy
        sampler_entropy = sampler_entropy.reshape(self.batch_size, self.num_samples, 1)

        if self.average_entropy:
            sampler_entropy = -sampler_entropy.mean(axis=1)
        else:
            sampler_entropy = -sampler_entropy[:, 0, :]

        stacked_s_batch = state_batch.repeat_interleave(self.top_action, dim=0)
        logp, _ = self.sampler.log_prob(stacked_s_batch, best_actions)
        sampler_loss = logp.reshape(self.batch_size, self.top_action, 1)
        sampler_loss = -1 * (sampler_loss.mean(axis=1) + self.tau * sampler_entropy).mean()

        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        self.sampler_optim.step()

