import copy

import numpy as np
import torch
import os
from src.agent.greedy_ac import GreedyAC
from src.agent.inac import InAC
from src.network.factory import init_policy_network, init_custom_network, init_optimizer
import src.network.torch_utils as torch_utils
from src.component.buffer import Buffer
from src.component.normalizer import init_normalizer


class GAC_InAC(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(GAC_InAC, self).__init__(cfg, average_entropy=average_entropy)
        if cfg.discrete_control:
            self.gac_a_dim = 1
            self.top_action = 1
            self.predict_action_encoder = init_normalizer("OneHot", self.env.action_space)
        else:
            self.predict_action_encoder = init_normalizer("Identity", None)

        self.predict_model = init_custom_network("Softmax", cfg.device, self.state_dim+self.action_dim, cfg.hidden_critic, 2,
                                                 cfg.activation, "None", cfg.layer_init_critic, cfg.layer_norm)
        self.predict_model_optim = init_optimizer(cfg.optimizer, list(self.predict_model.parameters()), cfg.lr_critic)

        inac_cfg = copy.deepcopy(cfg)
        inac_cfg.tau = 0.1
        self.safe_action_model = InAC(inac_cfg)

        self.optimal_reward = 1
        self.relax = 0.02
        self.start_safty_check = 1000

    def inner_update(self, trunc=False):
        data = super(GAC_InAC, self).inner_update(trunc=trunc)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']

        """ Update reward model """
        x = torch.concat((state_batch, self.predict_action_encoder(action_batch)), dim=1)
        _, logp, _ = self.predict_model(x)
        r_loss = -logp.mean()
        self.predict_model_optim.zero_grad()
        r_loss.backward()
        self.predict_model_optim.step()

        """ Update action model """
        self.safe_action_model.update()

    def step(self):
        observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
        action_tensor, _, pi_info = self.get_policy(observation_tensor,
                                                    with_grad=False, debug=self.cfg.debug)

        """predict the success for the given policy"""
        safe, _, info = self.safty_check(observation_tensor, action_tensor)
        if not safe:
            action_tensor, _, _ = self.safe_action_model.get_policy(observation_tensor, with_grad=False)

        action = torch_utils.to_np(action_tensor)[0]
        next_observation, reward, terminated, trunc, env_info = self.env.step(action)
        reset, truncate = self.update_stats(reward, terminated, trunc)

        """When using the known optimal action, do not add data to buffer"""
        self.buffer.feed([self.observation, action, reward, next_observation, int(terminated), int(truncate)])
        self.safe_action_model.buffer.feed([self.observation, action, reward, next_observation,
                                                   int(terminated), int(truncate)])

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
