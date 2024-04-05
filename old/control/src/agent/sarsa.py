import torch
import numpy as np
import os
import pickle as pkl
import time
import random
from scipy.stats import uniform

from src.agent.eval import Evaluation
from src.network.factory import init_critic_network, init_optimizer
from src.component.buffer import Buffer
import src.network.torch_utils as torch_utils



class Sarsa(Evaluation):
    def __init__(self, cfg):
        super(Sarsa, self).__init__(cfg)
        self.rng = np.random.RandomState(cfg.seed)

        # Continuous control initialization
        if cfg.discrete_control:
            # self.action_dim = self.env.action_space.n
            self.critic = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                              cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.critic_target = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                     cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete
        
        else:        
            # self.action_dim = np.prod(self.env.action_space.shape)
            self.critic = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_critic, 1,
                                              cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.critic_target = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_critic, 1,
                                                     cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.get_q_value = self.get_q_value_continuous
            self.get_q_value_target = self.get_q_value_target_continuous

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = init_optimizer(cfg.optimizer, list(self.critic.parameters()), cfg.lr_critic)
        
        self.buffer = Buffer(cfg.buffer_size, cfg.batch_size, cfg.seed)
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
        self.grid_size = 100
    
        
        if cfg.greedification == 'epsilon-greedy':
            self.epsilon = 0.1
            self.exploration_dist = uniform
            self.get_policy = self.get_epsilon_greedy_action
        else:
            raise NotImplementedError
        
        
        self.action_grid = self.get_action_grid() # initialize the action grid

        # Visit count heatmap
        if self.cfg.debug:
            action_cover_space, heatmap_shape = self.get_action_samples()
            self.visit_counts = [[0 for i in range(heatmap_shape[1])] for j in range(heatmap_shape[0])]
            self.x_action_increment = 10 / heatmap_shape[1]
            self.y_action_increment = 10 / heatmap_shape[0]

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
            action_tensor, _, pi_info = self.get_policy(observation_tensor, with_grad=False, debug=self.cfg.debug)
            action = torch_utils.to_np(action_tensor) # move the zero index to env_step.
            last_state = self.observation
            self.observation, reward, done, truncate, env_info = self.env_step(action)
            reset, truncate = self.update_stats(reward, done, truncate)
            self.buffer.feed([last_state, action[0], reward, self.observation, int(done), int(truncate)])

            i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
            self.info_log.append(i_log)

            if self.cfg.render:
                self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"])
            else:
                env_info.pop('interval_log', None)
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
        self.total_steps += 1
        self.ep_step += 1
        reset = False
        truncate = trunc or (self.ep_step == self.timeout)
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
        action_tensor, _,  pi_info = self.get_policy(observation_tensor,
                                                    with_grad=False, debug=self.cfg.debug)
        
        action = torch_utils.to_np(action_tensor) # move the zero index to env_step
        next_observation, reward, terminated, trunc, env_info = self.env_step(action)
        reset, truncate = self.update_stats(reward, terminated, trunc)
        self.buffer.feed([self.observation, action[0], reward, next_observation, int(terminated), int(truncate)])

        i_log = self.agent_debug_info(observation_tensor, action, pi_info, env_info)
        self.info_log.append(i_log)

        if self.cfg.render:
            self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"])
        else:
            env_info.pop('interval_log', None)
            i_log['critic_info'].pop('Q-function', None)

        self.update(trunc)

        if reset:
            next_observation, info = self.env_reset()
        self.observation = next_observation

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return
    
    
    # def decoupled_step(self, change_action, get_observation, do_update):
    #     """
    #     Similar to step, but allows the agent to different frequencies for choosing new actions, observing state and updating
    #     """
        
    #     if change_action: # if it is time change the action
    #         print('changing action')
    #         print(self.observation.shape)
    #         observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
    #         action_tensor, _, pi_info = self.get_policy(observation_tensor,
    #                                                     with_grad=False, debug=self.cfg.debug)
    #         action = torch_utils.to_np(action_tensor)[0]
    #         self.take_action(action) # take action in the environment
    #         self.last_action = action
    #         print('Choose action: {}'.format(action))

    #     if get_observation:
    #         print('getting observation')
    #         next_observation, reward, terminated, trunc, env_info = self.get_observation(action)
    #         reset, truncate = self.update_stats(reward, terminated, trunc)
    #         self.buffer.feed([self.observation, self.last_action, reward, next_observation, int(terminated), int(truncate)])
    #         if reset:
    #             next_observation, info = self.env_reset()

    #         self.observation = next_observation
        
    #     if do_update:
    #         print('updating')
    #         self.decoupled_update() # we use a different update funciton that does 
    #         # if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
    #         #     self.sync_target()
    
    #     # i_log = self.agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
    #     # self.info_log.append(i_log)
    #     # if self.cfg.render:
    #     #     self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"])
    #     # else:
    #     #     env_info.pop('interval_log', None)
    #     #     i_log['critic_info'].pop('Q-function', None)

    #     return

    # # actor-critic
    
    
   
    def get_action_grid(self):
        def cartesian_product(*arrays):
            # from https://stackoverflow.com/a/11146645
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[...,i] = a

            return arr.reshape(-1, la)
        
        grid_increment = 1/self.grid_size
        l = [np.arange(0, 1, grid_increment) for _ in range(self.action_dim)]
        grid = cartesian_product(*l)
        
        # grid = np.expand_dims(grid, axis=1) # this is necessary to concatenate actions with observations
        grid = torch_utils.tensor(grid, self.device)        
        return grid
    
    # def get_greedy_action(self, observation):
    #     t = time.time()
    #     # searches over action grid
    #     greedy_action = None
    #     max_q = -np.inf
    #     with torch.no_grad():
    #         for action in self.action_grid:
    #             x = torch.concat((observation, action), dim=1)
    #             q = self.critic(x)
    #             if q > max_q:
    #                 max_q = q
    #                 greedy_action = action
    #     print(time.time()-t)
    #     print(greedy_action)
    #     return greedy_action
    
    
    def get_greedy_action(self, observation):
        o =  torch.unsqueeze(observation, 0)
        obs_repeat = o.repeat(len(self.action_grid), 1)
        x = torch.concat((obs_repeat, self.action_grid), axis=1)
        x = torch_utils.tensor(x, self.device)

        q, _ = self.critic(x)
        max_q_idx = torch.argmax(q)
        greedy_action  = self.action_grid[max_q_idx, :]

        return greedy_action
    
    
    def get_epsilon_greedy_action(self, observation, with_grad, debug=False):  
        num_obs = observation.shape[0]
        actions = torch.zeros(num_obs, self.action_dim)
  
        for i, o in enumerate(observation):
            if random.random() <= self.epsilon:
                action = torch_utils.tensor(np.random.rand(1, self.action_dim), self.device)
            else:
                action = self.get_greedy_action(o)      
            actions[i, :] = action
            
        return actions, 0, {} # returning 0 for log_p and an empy info dict


    # Continuous control
    def get_q_value_continuous(self, observation, action, with_grad):
        x = torch.concat((observation, action), dim=1)
        if with_grad:
            q, _ = self.critic(x)
        else:
            with torch.no_grad():
                q, _ = self.critic(x)
        return q, None

    # Discrete control
    def get_q_value_discrete(self, observation, action, with_grad):
        action = action.squeeze(-1)
        if with_grad:
            qs, _ = self.critic(observation)
            q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        else:
            with torch.no_grad():
                qs, _ = self.critic(observation)
                q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        return q, qs

    def get_q_value_target_continuous(self, observation, action):
        x = torch.concat((observation, action), dim=1)
        with torch.no_grad():
            q, _ = self.critic_target(x)
        return q, None

    def get_q_value_target_discrete(self, observation, action):
        action = action.squeeze(-1)
        with torch.no_grad():
            qs, _ = self.critic_target(observation)
            q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        return q, qs

    def get_v_value(self, observation, with_grad):
        if with_grad:
            v = self.v_baseline(observation)
        else:
            with torch.no_grad():
                v = self.v_baseline(observation)
        return v

    def get_data(self):
        states, actions, rewards, next_states, terminals, truncations = self.buffer.sample()
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

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']
        
        # critic update
        reward_batch = torch.clamp(reward_batch, -2, 1)
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)
        
        target = reward_batch + mask_batch * self.gamma * next_q
        _, q_ens = self.get_q_value(state_batch, action_batch, with_grad=True)
       

        q_loss = self.ensemble_mse(target, q_ens) #torch.nn.functional.mse_loss(target, q_value)
        self.critic_optimizer.zero_grad()
        self.ensemble_critic_loss_backward(q_loss) #q_loss.backward()
        self.critic_optimizer.step()

       
       
       
       
    
    def save(self):
        parameters_dir = self.parameters_dir
        
        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.critic.state_dict(), path)
    
        path = os.path.join(parameters_dir, "critic_target")
        torch.save(self.critic_target.state_dict(), path)
    
        path = os.path.join(parameters_dir, "critic_opt")
        torch.save(self.critic_optimizer.state_dict(), path)
        
        path = os.path.join(parameters_dir, "buffer.pkl")
        with open(path, "wb") as f:
            pkl.dump(self.buffer, f)

    def savevis(self):
        if self.cfg.render:
            self.save_render(os.path.join(self.cfg.vis_path))

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
            self.logger.info("Load actor actor_optimizer from {}".format(pth))
        
            pth = os.path.join(parameters_dir, 'critic_opt')
            self.critic_optimizer.load_state_dict(torch.load(pth, map_location=self.device))
            self.logger.info("Load critic actor_optimizer from {}".format(pth))

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
            action_cover_space, heatmap_shape = self.get_action_samples(n=50)
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
