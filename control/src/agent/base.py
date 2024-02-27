import torch
import numpy as np
import os
import pickle as pkl
import time

from src.agent.eval import Evaluation
from src.network.factory import init_policy_network, init_critic_network, init_optimizer
# from src.component.normalizer import init_normalizer
from src.component.buffer import init_buffer
import src.network.torch_utils as torch_utils



class BaseAC(Evaluation):
    def __init__(self, cfg):
        super(BaseAC, self).__init__(cfg)
        self.rng = np.random.RandomState(cfg.seed)
        # Continuous control initialization
        if cfg.discrete_control:
            # self.action_dim = self.env.action_space.n
            self.actor = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                             cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                             cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
            self.critic = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                              cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.critic_target = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                     cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete
            self.random_policy = init_policy_network("UniformRandomDisc", cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                     cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                                     cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm
                                                     )
        else:        
            # self.action_dim = np.prod(self.env.action_space.shape)
            self.actor = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                             cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                             cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
            self.critic = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_critic, 1,
                                              cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.critic_target = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_critic, 1,
                                                     cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.get_q_value = self.get_q_value_continuous
            self.get_q_value_target = self.get_q_value_target_continuous
            self.random_policy = init_policy_network("UniformRandomCont", cfg.device, self.state_dim, cfg.hidden_critic,
                                                     self.action_dim,
                                                     cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                                     cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm
                                                     )

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = init_optimizer(cfg.optimizer, list(self.actor.parameters()), cfg.lr_actor)
        self.critic_optimizer = init_optimizer(cfg.optimizer, self.critic.parameters(independent=True), cfg.lr_critic, ensemble=True)

        # self.buffer = Buffer(cfg.buffer_size, cfg.batch_size, cfg.seed)
        self.buffer = init_buffer(cfg.buffer_type, cfg)
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.parameters_dir = cfg.parameters_path
        self.polyak = cfg.polyak
        self.use_target_network = cfg.use_target_network
        self.target_network_update_freq = 1
        self.cfg = cfg
        self.discrete_control = cfg.discrete_control
        self.decision_freq = cfg.decision_freq
        self.last_action = None 

        # Visit count heatmap
        if self.cfg.debug:
            action_cover_space, heatmap_shape = self.get_action_samples()
            self.visit_counts = [[0 for i in range(heatmap_shape[1])] for j in range(heatmap_shape[0])]
            self.x_action_increment = 1. / heatmap_shape[1]
            self.y_action_increment = 1. / heatmap_shape[0]

    def fill_buffer(self, online_data_size):
        track_states = []
        track_actions = []
        track_rewards = []
        track_next_states = []
        track_done = []
        track_return = []
        track_step = []
        self.observation, info = self.env_reset()
       
        ep_steps = 0
        ep_return = 0
        done = False
        for _ in range(online_data_size):
            observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
            # action_tensor, _, pi_info = self.get_policy(observation_tensor, with_grad=False, debug=self.cfg.debug)
            with torch.no_grad():
                action_tensor, _, pi_info = self.random_policy(observation_tensor, self.cfg.debug)
            action = torch_utils.to_np(action_tensor) # move the zero index to env_step.
            last_state = self.observation
            self.observation, reward, done, truncate, env_info = self.env_step(action)
            reset, truncate = self.update_stats(reward, done, truncate)
            self.buffer.feed([last_state, action[0], reward, self.observation, int(done), int(truncate)])

            if self.cfg.debug:
                i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
                self.info_log.append(i_log)

                if self.cfg.render:
                    self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"],
                                env_info['environment_pid'])
                else:
                    env_info.pop('interval_log', None)
                    env_info.pop('environment_pid', None)
                    i_log['critic_info'].pop('Q-function', None)

            track_states.append(last_state)
            track_actions.append(action[0])
            track_rewards.append(reward)
            track_next_states.append(self.observation)
            track_done.append(done)
            ep_return += reward
            ep_steps += 1
            if reset:
                self.observation, info = self.env_reset()
                track_return.append(ep_return)
                track_step.append(ep_steps)
                ep_steps = 0
                ep_return = 0
                done = False
        self.logger.info('Fill {} transitions to buffer. Length of buffer is {}'.format(online_data_size, self.buffer.size))
        avg = np.array(track_return).mean() if len(track_return) > 0 else np.nan
        self.logger.info('Averaged return of filled data is {:.4f}'.format(avg))
        return np.array(track_states), np.array(track_actions), np.array(track_rewards), np.array(track_next_states), np.array(track_done), track_return, track_step

    def update_stats(self, reward, done, trunc):
        self.ep_reward += reward
        self.step_rewards.append(reward)
        self.total_steps += 1
        self.ep_step += 1
        reset = False
        truncate = trunc or (self.ep_step == self.timeout)
        if not done and self.timeout >= self.cfg.max_steps:
            if self.ep_step and self.ep_step % self.reward_window == 0:
                self.ep_returns.append(np.array(self.step_rewards[max(0, self.total_steps-self.reward_window): ]).sum())
        if done or truncate:
            self.ep_returns.append(self.ep_reward)
            self.num_episodes += 1
            if self.evaluation_criteria == "return":
                self.add_train_log(self.ep_reward)
            elif self.evaluation_criteria == "steps":
                self.add_train_log(self.ep_step)
            else:
                raise NotImplementedError
            self.ep_reward = 0
            self.ep_step = 0
            reset = True
        return reset, truncate

    def step(self):
        observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
        action_tensor, _, pi_info = self.get_policy(observation_tensor,
                                                    with_grad=False, debug=self.cfg.debug)
        action = torch_utils.to_np(action_tensor) # move the zero index to env_step.
        next_observation, reward, terminated, trunc, env_info = self.env_step(action)
        reset, truncate = self.update_stats(reward, terminated, trunc)
        self.buffer.feed([self.observation, action[0], reward, next_observation, int(terminated), int(truncate)])

        if self.cfg.debug:
            i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
            self.info_log.append(i_log)

            if self.cfg.render:
                self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"],
                            env_info['environment_pid'])
            else:
                env_info.pop('interval_log', None)
                env_info.pop('environment_pid', None)
                i_log['critic_info'].pop('Q-function', None)

        self.update(trunc)

        if reset:
            next_observation, info = self.env_reset()
        self.observation = next_observation

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        if self.cfg.buffer_type == "Prioritized":
            data = self.get_all_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data[
                'reward'], data['obs2'], 1 - data['done']

            next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
            next_q, _ = self.get_q_value_target(next_state_batch, next_action)
            target = reward_batch + mask_batch * self.gamma * next_q
            q_value, _ = self.get_q_value(state_batch, action_batch, with_grad=True)
            priority = torch.abs(target - q_value).squeeze(-1)
            priority = torch_utils.to_np(priority)
            self.buffer.update_priorities(priority)
        return
    
    
    def decoupled_step(self, change_action, get_observation, do_update):
        """
        Similar to step, but allows the agent to different frequencies for choosing new actions, observing state and updating
        """
        
        if change_action: # if it is time change the action
            print('changing action')
            print(self.observation.shape)
            observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
            action_tensor, _, pi_info = self.get_policy(observation_tensor,
                                                        with_grad=False, debug=self.cfg.debug)
            action = torch_utils.to_np(action_tensor)[0]
            self.take_action(action) # take action in the environment
            self.last_action = action
            print('Choose action: {}'.format(action))

        if get_observation:
            print('getting observation')
            next_observation, reward, terminated, trunc, env_info = self.get_observation(action)
            reset, truncate = self.update_stats(reward, terminated, trunc)
            self.buffer.feed([self.observation, self.last_action, reward, next_observation, int(terminated), int(truncate)])
            if reset:
                next_observation, info = self.env_reset()
            self.observation = next_observation
        
        if do_update:
            print('updating')
            self.decoupled_update() # we use a different update funciton that does 
            # if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            #     self.sync_target()
    
        # i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
        # self.info_log.append(i_log)
        # if self.cfg.render:
        #     self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"])
        # else:
        #     env_info.pop('interval_log', None)
        #     i_log['critic_info'].pop('Q-function', None)

        return

    # actor-critic
    def get_policy(self, observation, with_grad, debug=False):
        if with_grad:
            action, logp, info = self.actor(observation, debug)
        else:
            with torch.no_grad():
                action, logp, info = self.actor(observation, debug)
        return action, logp.unsqueeze(-1), info

    # Continuous control
    def get_q_value_continuous(self, observation, action, with_grad):
        x = torch.concat((observation, action), dim=1)
        if with_grad:
            q, qs = self.critic(x)
        else:
            with torch.no_grad():
                q, qs = self.critic(x)
        return q, qs

    # Discrete control
    def get_q_value_discrete(self, observation, action, with_grad):
        action = self.action_normalizer.denormalize(action)
        action = action.squeeze(-1)
        if with_grad:
            qs, qsens = self.critic(observation)
            q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        else:
            with torch.no_grad():
                qs, qsens = self.critic(observation)
                q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        qens = []
        for i in range(self.cfg.critic_ensemble):
            qens.append(qsens[i][np.arange(len(action)), action.long()].unsqueeze(-1))
        return q, qens

    def get_q_value_target_continuous(self, observation, action):
        x = torch.concat((observation, action), dim=1)
        with torch.no_grad():
            q, qs = self.critic_target(x)
        return q, qs

    def get_q_value_target_discrete(self, observation, action):
        action = self.action_normalizer.denormalize(action)
        action = action.squeeze(-1)
        with torch.no_grad():
            qs, qsens = self.critic_target(observation)
            q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        qens = []
        for i in range(self.cfg.critic_ensemble):
            qens.append(qsens[i][np.arange(len(action)), action.long()].unsqueeze(-1))
        return q, qens

    def get_v_value(self, observation, with_grad):
        if with_grad:
            v = self.v_baseline(observation)
        else:
            with torch.no_grad():
                v = self.v_baseline(observation)
        return v

    def get_data(self, batch_size=None):
        states, actions, rewards, next_states, terminals, truncations = self.buffer.sample(batch_size)
        in_ = torch_utils.tensor(states, self.device)
        actions = torch_utils.tensor(actions, self.device)
        r = torch_utils.tensor(rewards, self.device)
        ns = torch_utils.tensor(next_states, self.device)
        d = torch_utils.tensor(terminals, self.device)
        t = torch_utils.tensor(truncations, self.device)
        data = {
            'obs': in_,
            'act': actions,
            'reward': r,
            'obs2': ns,
            'done': d,
            'trunc': t,
        }
        return data

    def get_all_data(self):
        """ load a batch """
        states, actions, rewards, next_states, terminals, truncations = self.buffer.sample_batch()
        in_ = torch_utils.tensor(states, self.device)
        actions = torch_utils.tensor(actions, self.device)
        r = torch_utils.tensor(rewards, self.device)
        ns = torch_utils.tensor(next_states, self.device)
        d = torch_utils.tensor(terminals, self.device)
        t = torch_utils.tensor(truncations, self.device)
        data = {
            'obs': in_,
            'act': actions,
            'reward': r,
            'obs2': ns,
            'done': d,
            'trunc': t,
        }
        return data

    def update(self, trunc=False):
        if self.total_steps % self.update_freq == 0:
            for _ in range(self.update_freq):
                self.inner_update(trunc)
                
    def decoupled_update(self):
        self.inner_update()

    # def decoupled_inner_update(self):
    #     pass
    #     # raise NotImplementedError

    def inner_update(self):
        pass
        # raise NotImplementedError

    def ensemble_mse(self, target, q_ens):
        mses = [torch.nn.functional.mse_loss(target, q) for q in q_ens]
        return mses

    def ensemble_critic_loss_backward(self, loss):
        for i in range(len(loss)):
            loss[i].backward(inputs=list(self.critic.parameters(independent=True)[i]))
        return

    def save(self):
        parameters_dir = self.parameters_dir
        
        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.critic.state_dict(), path)
    
        path = os.path.join(parameters_dir, "critic_target")
        torch.save(self.critic_target.state_dict(), path)
    
        path = os.path.join(parameters_dir, "critic_opt")
        torch.save(self.critic_optimizer.state_dict(), path)
        
        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.actor.state_dict(), path)
    
        path = os.path.join(parameters_dir, "actor_opt")
        torch.save(self.actor_optimizer.state_dict(), path)
        
        path = os.path.join(parameters_dir, "buffer.pkl")
        with open(path, "wb") as f:
            pkl.dump(self.buffer, f)

    def savevis(self):
        if self.cfg.render:
            self.save_render(os.path.join(self.cfg.vis_path))

    def clean(self):
        path = self.cfg.parameters_path + "/buffer.pkl"
        if os.path.isfile(path):
            os.remove(path)
        prefixed = [filename for filename in os.listdir(self.cfg.parameters_path) if filename.startswith("prefill_")]
        for filename in prefixed:
            path = os.path.join(self.cfg.parameters_path, filename)
            if os.path.isfile(path):
                os.remove(path)

    def load(self, parameters_dir, checkpoint=False):
        pth = os.path.join(parameters_dir, 'actor_net')
        self.actor.load_state_dict(torch.load(pth, map_location=self.device))
        self.logger.info("Load actor function from {}".format(pth))
    
        pth = os.path.join(parameters_dir, 'critic_net')
        self.critic.load_state_dict(torch.load(pth, map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.logger.info("Load critic function from {}".format(pth))

        if checkpoint:
            pth = os.path.join(parameters_dir, 'actor_opt')
            self.actor_optimizer.load_state_dict(torch.load(pth, map_location=self.device))
            self.logger.info("Load actor optimizer from {}".format(pth))
        
            pth = os.path.join(parameters_dir, 'critic_opt')
            self.critic_optimizer.load_state_dict(torch.load(pth, map_location=self.device))
            self.logger.info("Load critic optimizer from {}".format(pth))

            pth = os.path.join(parameters_dir, 'critic_target')
            self.critic_target.load_state_dict(torch.load(pth, map_location=self.device))
            self.logger.info("Load critic target from {}".format(pth))

            path = os.path.join(parameters_dir, "buffer.pkl")
            with open(path, "rb") as f:
                self.buffer = pkl.load(f)

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def agent_debug_info(self, observation_tensor, action_tensor, pi_info, env_info):
        if self.cfg.debug:
            # Update Action Visit Counts
            action = torch_utils.to_np(action_tensor)[0]
            if self.discrete_control:
                x_action_ind = 0
                y_action_ind = self.action_normalizer.denormalize(action.reshape((-1, self.action_dim)))[0,0]
            else:
                x_action_ind = int(action[0] / self.x_action_increment)
                y_action_ind = int(action[1] / self.y_action_increment)
                if x_action_ind == 10 / self.x_action_increment:
                    x_action_ind -= 1
                if y_action_ind == 10 / self.y_action_increment:
                    y_action_ind -= 1
            self.visit_counts[y_action_ind][x_action_ind] += 1
        
            # Update Q heatmap
            q_current, _ = self.get_q_value(observation_tensor, action_tensor, with_grad=False)
            q_current = torch_utils.to_np(q_current)
            action_cover_space, heatmap_shape = self.get_action_samples(n=20)
            stacked_o = observation_tensor.repeat_interleave(len(action_cover_space), dim=0)
            action_cover_space_tensor = torch_utils.tensor(action_cover_space, self.device)
            q_cover_space, _ = self.get_q_value(stacked_o, action_cover_space_tensor, with_grad=False)
            q_cover_space = torch_utils.to_np(q_cover_space)
            q_cover_space = q_cover_space.reshape(heatmap_shape)
            coord = np.array([action_cover_space[:, d].reshape(heatmap_shape) for d in range(action_cover_space.shape[1])])
            if heatmap_shape[1] == 1:
                coord = np.concatenate((coord, np.zeros(coord.shape)), axis=0)
            i_log = {
                "actor_info": pi_info,
                "critic_info": {'Q': q_current,
                                'Q-function': [q_cover_space, coord]},
                "env_info": env_info,
                "action_visits": {'sum': np.array(self.visit_counts),
                                  'curr_action': (x_action_ind, y_action_ind)}
            }
        else:
            i_log = {
                "actor_info": pi_info,
                "critic_info": {},
                "env_info": env_info
            }
        return i_log

class BaseValue(BaseAC):
    def __init__(self, cfg):
        super(BaseValue, self).__init__(cfg)
        assert self.discrete_control
        self.exploration = cfg.exploration

    def get_policy(self, observation, with_grad, debug=False):
        if with_grad or self.rng.random() >= self.exploration:
            qs, _ = self.critic(observation)
            a = torch.argmax(qs, dim=1, keepdim=True)
        else:
            a = self.rng.randint(self.action_dim, size=len(observation)).reshape((-1, 1))
            a = torch.tensor(a, dtype=torch.int8).to(self.device)
        return a, None, None
