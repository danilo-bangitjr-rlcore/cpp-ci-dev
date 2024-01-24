import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.agent.greedy_ac import GreedyAC, GreedyACDiscrete
from src.network.factory import init_policy_network, init_critic_network, init_optimizer, init_custom_network
import src.network.torch_utils as torch_utils
from src.network.torch_utils import clone_model_0to1, clone_gradient, move_gradient_to_network


class LineSearchAgent(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(LineSearchAgent, self).__init__(cfg)

        # Leave positions for backup the networks
        if self.discrete_control:
            self.gac_a_dim = 1
            self.top_action = 1
            self.actor_copy = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                                  cfg.beta_parameter_bias, cfg.activation,
                                                  cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
            self.critic_copy = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                   cfg.activation, cfg.layer_init_critic, cfg.layer_norm)
        else:
            self.actor_copy = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                                  cfg.beta_parameter_bias, cfg.activation,
                                                  cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
            self.critic_copy = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_critic, 1,
                                                   cfg.activation, cfg.layer_init_critic, cfg.layer_norm)

        self.actor_opt_copy = init_optimizer(cfg.optimizer, list(self.actor.parameters()), cfg.lr_actor)
        self.critic_opt_copy = init_optimizer(cfg.optimizer, list(self.critic.parameters()), cfg.lr_critic)

        self.actor_lr_weight = 1  # always start with 1, tend to use a large learning rate initialization (cfg.lr_critic)
        self.actor_lr_weight_copy = 1
        self.critic_lr_weight = 1
        self.critic_lr_weight_copy = 1

        # Networks for uncertainty measure
        self.ftrue0 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.ftrue1 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus0 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus1 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)

        # Ensure the nonlinear net between learning net and target net are the same
        clone_model_0to1(self.ftrue0.random_network, self.fbonus0.random_network)
        clone_model_0to1(self.ftrue1.random_network, self.fbonus1.random_network)

        # optimizer for learning net only
        self.bonus_opt_0 = init_optimizer(cfg.optimizer, list(self.fbonus0.parameters()), cfg.lr_critic)
        self.bonus_opt_1 = init_optimizer(cfg.optimizer, list(self.fbonus1.parameters()), cfg.lr_critic)

        # TODO: These should go to variable in main function later
        self.max_backtracking = 10
        self.reducing_rate = 0.5
        self.increasing_rate = 1.1 # Not sure if this is going to be used
        assert self.cfg.batch_size >= 256
        assert self.max_backtracking > 0

        self.random_prefill()

    def random_prefill(self):
        pth = self.cfg.parameters_path + "/prefill_{}.pkl".format(self.cfg.etc_buffer_prefill)
        if not os.path.isfile(pth):
            '''using etc's parameter here'''
            random_policy = init_policy_network("Beta", self.cfg.device, self.state_dim, [], self.action_dim,
                                                0, self.cfg.activation, self.cfg.head_activation, "Const/1/0", False)
            reset = True
            for t in range(self.cfg.etc_buffer_prefill):
                if t % 1000 == 0:
                    print("Prefill t={}".format(t))
                if reset:
                    observation, info = self.env_reset()
                observation_tensor = torch_utils.tensor(observation.reshape((1, -1)), self.device)
                with torch.no_grad():
                    action, _, _ = random_policy(observation_tensor, False)
                    action = torch_utils.to_np(action)
                next_observation, reward, terminated, trunc, env_info = self.env_step(action)
                reset = terminated or trunc
                self.buffer.feed([observation, action[0], reward, next_observation, int(terminated), int(trunc)])
                observation = next_observation
            with open(pth, "wb") as f:
                pkl.dump(self.buffer, f)
        else:
            print("Load prefilled data")
            with open(pth, "rb") as f:
                self.buffer = pkl.load(f)

        # # For debugging
        # plt.figure()
        # data = self.buffer.get_all_data()
        # act = np.array([i[1] for i in data])
        # rwd = np.array([i[2] for i in data])
        # plt.scatter(act[:, 0], act[:, 1], c=rwd, s=2)
        # plt.show()

    def explore_bonus_update(self, state, action, reward, next_state, next_action, mask):
        in_ = torch.concat((state, action), dim=1)
        in_p1 = torch.concat((next_state, next_action), dim=1)
        with torch.no_grad():
            true0_t, _ = self.ftrue0(in_)
            true1_t, _ = self.ftrue1(in_)
            true0_tp1, _ = self.ftrue0(in_p1)
            true1_tp1, _ = self.ftrue1(in_p1)
        reward0 = true0_t - mask * self.gamma * true0_tp1
        reward1 = true1_t - mask * self.gamma * true1_tp1

        pred0, _ = self.fbonus0(in_)
        pred1, _ = self.fbonus1(in_)
        with torch.no_grad():
            target0 = reward0 + mask * self.gamma * self.fbonus0(in_p1)[0]
            target1 = reward1 + mask * self.gamma * self.fbonus1(in_p1)[0]
        loss0 = nn.functional.mse_loss(pred0, target0)
        loss1 = nn.functional.mse_loss(pred1, target1)

        self.bonus_opt_0.zero_grad()
        loss0.backward()
        self.bonus_opt_0.step()
        self.bonus_opt_1.zero_grad()
        loss1.backward()
        self.bonus_opt_1.step()

    def explore_bonus_eval(self, state, action):
        in_ = torch.concat((state, action), dim=1)
        with torch.no_grad():
            pred0, _ = self.fbonus0(in_)
            pred1, _ = self.fbonus1(in_)
            true0, _ = self.ftrue0(in_)
            true1, _ = self.ftrue1(in_)
        b, _ = torch.max(torch.concat([torch.abs(pred0 - true0), torch.abs(pred1 - true1)], dim=1), dim=1)
        return b.detach()

    def eval_error_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_q):
        q, _ = self.get_q_value(state_batch, action_batch, with_grad=False)
        target = reward_batch + mask_batch * self.gamma * next_q
        error = nn.functional.mse_loss(q.detach(), target.detach())
        return error

    def eval_error_actor(self, state_batch):
        _, logp, _ = self.get_policy(state_batch, with_grad=False)
        return -torch.exp(logp.mean()).detach()

    def parameter_backup_critic(self):
        clone_model_0to1(self.critic, self.critic_copy)
        clone_model_0to1(self.critic_optimizer, self.critic_opt_copy)
        self.critic_lr_weight_copy = self.critic_lr_weight

    def parameter_backup_actor(self):
        clone_model_0to1(self.actor, self.actor_copy)
        clone_model_0to1(self.actor_optimizer, self.actor_opt_copy)
        self.actor_lr_weight_copy = self.actor_lr_weight

    def undo_update_critic(self):
        clone_model_0to1(self.critic_copy, self.critic)
        clone_model_0to1(self.critic_opt_copy, self.critic_optimizer)

    def undo_update_actor(self):
        clone_model_0to1(self.actor_copy, self.actor)
        clone_model_0to1(self.actor_opt_copy, self.actor_optimizer)

    def reset_critic(self):
        self.cfg.lr_critic *= 0.5
        if self.cfg.discrete_control:
            self.critic = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim, self.cfg.hidden_critic, self.action_dim,
                                              self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm)
            self.critic_target = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim, self.cfg.hidden_critic, self.action_dim,
                                                     self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm)
        else:
            self.critic = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim + self.action_dim, self.cfg.hidden_critic, 1,
                                              self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm)
            self.critic_target = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim + self.action_dim, self.cfg.hidden_critic, 1,
                                                     self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm)

        clone_model_0to1(self.critic, self.critic_target)
        self.critic_optimizer = init_optimizer(self.cfg.optimizer, list(self.critic.parameters()), self.cfg.lr_critic)
        print("Critic reset, learning rate", self.cfg.lr_critic)
        self.critic_lr_weight = 1
        self.critic_lr_weight_copy = 1

        for _ in range(int(np.ceil(self.buffer.size / self.cfg.batch_size))*10):
            data = self.get_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                data['obs2'], 1 - data['done']
            q_loss, _ = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            # do not change lr_weight here
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()

    def reset_actor(self):
        self.cfg.lr_actor *= 0.5
        self.actor = init_policy_network(self.cfg.actor, self.cfg.device, self.state_dim, self.cfg.hidden_actor,
                                         self.action_dim, self.cfg.beta_parameter_bias, self.cfg.activation,
                                         self.cfg.head_activation, self.cfg.layer_init_actor, self.cfg.layer_norm)
        self.actor_optimizer = init_optimizer(self.cfg.optimizer, list(self.actor.parameters()), self.cfg.lr_actor)
        print("Actor reset, learning rate", self.cfg.lr_actor)
        self.actor_lr_weight = 1
        self.actor_lr_weight_copy = 1

        for _ in range(int(np.ceil(self.buffer.size / self.cfg.batch_size))*10):
            data = self.get_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                data['obs2'], 1 - data['done']
            _, _, _, _, stacked_s_batch, best_actions, logp = self.actor_loss(state_batch)
            pi_loss = (-logp + self.explore_bonus_eval(stacked_s_batch, best_actions)).mean()
            # do not change lr_weight here
            self.actor_optimizer.zero_grad()
            pi_loss.backward()
            self.actor_optimizer.step()

    def backtrack_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)
        before_error = self.eval_error_critic(state_batch, action_batch, reward_batch, mask_batch, next_q)
        q_loss, next_action = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        q_loss_weighted = q_loss * self.critic_lr_weight # The weight is supposed to always be 1.
        self.critic_optimizer.zero_grad()
        q_loss_weighted.backward()
        grad_rec_critic = clone_gradient(self.critic)

        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                print("backtrack critic", bi)
                self.critic_optimizer.zero_grad()
                move_gradient_to_network(self.critic, grad_rec_critic, self.critic_lr_weight)
            self.critic_optimizer.step()
            after_error = self.eval_error_critic(state_batch, action_batch, reward_batch, mask_batch, next_q)

            if after_error > before_error and bi < self.max_backtracking-1:
                self.critic_lr_weight *= 0.5
                self.undo_update_critic()
            elif after_error > before_error and bi == self.max_backtracking-1:
                print("====", after_error, before_error, self.critic_lr_weight, self.cfg.lr_critic)
                self.reset_critic()
            else:
                break
        self.critic_lr_weight = self.critic_lr_weight_copy
        return next_action

    def backtrack_actor(self, state_batch):
        before_error = self.eval_error_actor(state_batch)
        _, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, logp = self.actor_loss(state_batch)
        pi_loss = (-logp + self.explore_bonus_eval(stacked_s_batch, best_actions)).mean()
        pi_loss_weighted = pi_loss * self.actor_lr_weight
        self.actor_optimizer.zero_grad()
        pi_loss_weighted.backward()
        grad_rec_actor = clone_gradient(self.actor)

        for bi in range(self.max_backtracking):
            if bi > 0:
                print("actor", bi)
                self.actor_optimizer.zero_grad()
                move_gradient_to_network(self.actor, grad_rec_actor, self.actor_lr_weight)
            self.actor_optimizer.step()
            after_error = self.eval_error_actor(state_batch)

            if after_error >= before_error and bi < self.max_backtracking-1:
                self.actor_lr_weight *= 0.5
                self.undo_update_actor()
            elif after_error >= before_error and bi == self.max_backtracking-1:
                self.reset_actor()
            else:
                break
        self.actor_lr_weight = self.actor_lr_weight_copy
        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']
        # critic update
        self.parameter_backup_critic()
        next_action = self.backtrack_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

        # uncertainty update
        self.explore_bonus_update(state_batch, action_batch, reward_batch, next_state_batch, next_action, mask_batch)

        # actor update
        # self.parameter_backup_actor()
        # repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions = self.backtrack_actor(state_batch)

        #GAC's actor update
        pi_loss, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, _ = self.actor_loss(state_batch)
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()


        # sampler update
        sampler_loss = self.proposal_loss(sample_actions, repeated_states,
                                          stacked_s_batch, best_actions, sorted_q, state_batch)
        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        self.sampler_optim.step()
