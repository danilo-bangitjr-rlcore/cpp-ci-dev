import torch
import numpy as np
import os
import pickle as pkl

from src.agent.eval import Evaluation
from src.network.factory import init_policy_network, init_critic_network, init_optimizer
from src.component.normalizer import init_normalizer
from src.component.buffer import Buffer
import src.network.torch_utils as torch_utils


class BaseAC(Evaluation):
    def __init__(self, cfg):
        super(BaseAC, self).__init__(cfg)
        self.rng = np.random.RandomState(cfg.seed)

        """
            parser.add_argument('--policy', default='Beta', type=str)
            parser.add_argument('--critic', default='FC', type=str)
            parser.add_argument('--optimizer', default='RMSProp', type=str)
            parser.add_argument('--state_normalizer', default='identity', type=str)
            parser.add_argument('--reward_normalizer', default='identity', type=str)
        """

        # Continuous control initialization
        self.actor = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_units, self.action_dim, cfg.action_scale, cfg.action_bias)

        if cfg.discrete_control:
            self.critic = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_units, self.action_dim)
            self.critic_target = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_units, self.action_dim)
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete
        else:
            self.critic = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_units, 1)
            self.critic_target = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_units, 1)
            self.get_q_value = self.get_q_value_continuous
            self.get_q_value_target = self.get_q_value_target_continuous
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = init_optimizer(cfg.optimizer, list(self.actor.parameters()), cfg.lr_actor)
        self.critic_optimizer = init_optimizer(cfg.optimizer, list(self.critic.parameters()), cfg.lr_critic)

        # self.constrain_weight = 0
        # self.lr_constrain = cfg.lr_constrain
        
        self.buffer = Buffer(cfg.buffer_size, cfg.batch_size, cfg.seed)
        self.batch_size = cfg.batch_size
        self.state_normalizer = init_normalizer(cfg.state_normalizer, {})
        self.reward_normalizer = init_normalizer(cfg.reward_normalizer, {})
        self.gamma = cfg.gamma
        self.parameters_dir = cfg.parameters_path
        self.polyak = cfg.polyak
        self.use_target_network = cfg.use_target_network
        self.target_network_update_freq = 1
        self.cfg = cfg
        self.discrete_control = cfg.discrete_control

    def fill_buffer(self, online_data_size):
        track_states = []
        track_actions = []
        track_rewards = []
        track_next_states = []
        track_done = []
        track_return = []
        track_step = []
        state, info = self.env.reset()
        ep_steps = 0
        ep_return = 0
        done = False
        for _ in range(online_data_size):
            if done:
                state = self.env.reset()
                track_return.append(ep_return)
                track_step.append(ep_steps)
                ep_steps = 0
                ep_return = 0
            action = self.eval_step(state.reshape((1, -1)))[0]
            last_state = state
            state, reward, done, truncate, _ = self.env.step(action)
            self.buffer.feed([last_state, action, reward, state, int(done), int(truncate)])
            track_states.append(last_state)
            track_actions.append(action)
            track_rewards.append(reward)
            track_next_states.append(state)
            track_done.append(done)
            ep_return += reward
            ep_steps += 1
            if done or ep_steps == self.timeout:
                done = True
        self.logger.info('Fill {} transitions to buffer. Length of buffer is {}'.format(online_data_size, self.buffer.size))
        avg = np.array(track_return).mean() if len(track_return) > 0 else np.nan
        self.logger.info('Averaged return of filled data is {:.4f}'.format(avg))
        return np.array(track_states), np.array(track_actions), np.array(track_rewards), np.array(track_next_states), np.array(track_done), track_return, track_step

    def update_stats(self, reward, done):
        self.ep_reward += reward
        self.total_steps += 1
        self.ep_step += 1
        reset = False
        truncate = self.ep_step == self.timeout
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
        action, _, _ = self.get_policy(torch_utils.tensor(self.observation.reshape((1, -1)), self.device), with_grad=False)
        action = torch_utils.to_np(action)[0]
        next_observation, reward, terminated, trunc, info = self.env.step(action)
        assert trunc == False
        reset, truncate = self.update_stats(reward, terminated)
        self.buffer.feed([self.observation, action, reward, next_observation, int(terminated), int(truncate)])
        
        if self.cfg.render:
            self.render(np.array(info['interval_log']))
    
        self.update()
        
        if reset:
            next_observation, info = self.env.reset()
        self.observation = next_observation

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return

    # actor-critic
    def get_policy(self, observation, with_grad):
        if with_grad:
            action, logp, dist = self.actor(observation)
        else:
            with torch.no_grad():
                action, logp, dist = self.actor(observation)
        return action, logp.unsqueeze(-1), dist

    # Continuous control
    def get_q_value_continuous(self, observation, action, with_grad):
        x = torch.concat((observation, action), dim=1)
        if with_grad:
            q = self.critic(x)
        else:
            with torch.no_grad():
                q = self.critic(x)
        return q, None

    # Discrete control
    def get_q_value_discrete(self, observation, action, with_grad):
        action = action.squeeze(-1)
        if with_grad:
            qs = self.critic(observation)
            q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        else:
            with torch.no_grad():
                qs = self.critic(observation)
                q = qs[np.arange(len(action)), action.long()].unsqueeze(-1)
        return q, qs

    def get_q_value_target_continuous(self, observation, action):
        x = torch.concat((observation, action), dim=1)
        with torch.no_grad():
            q = self.critic_target(x)
        return q, None

    def get_q_value_target_discrete(self, observation, action):
        action = action.squeeze(-1)
        with torch.no_grad():
            qs = self.critic_target(observation)
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
        in_ = torch_utils.tensor(self.state_normalizer(states), self.device)
        actions = torch_utils.tensor(actions, self.device)
        r = torch_utils.tensor(self.reward_normalizer(rewards), self.device)
        ns = torch_utils.tensor(self.state_normalizer(next_states), self.device)
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

    def update(self):
        if self.total_steps % self.update_freq == 0:
            for _ in range(self.update_freq):
                self.inner_update()

    def inner_update(self):
        raise NotImplementedError
    
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


class BaseValue(BaseAC):
    def __init__(self, cfg):
        super(BaseValue, self).__init__(cfg)
        assert self.discrete_control
        self.exploration = cfg.exploration

    def get_policy(self, observation, with_grad):
        if eval or self.rng.random() >= self.exploration:
            qs, _ = self.critic(observation)
            a = torch.argmax(qs, dim=1, keepdim=True)
        else:
            a = self.rng.randint(self.action_dim, size=len(observation)).reshape((-1, 1))
            a = torch.tensor(a, dtype=torch.int8).to(self.device)
        return a, None, None
