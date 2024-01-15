import copy
import os
import imageio
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gymnasium.spaces.utils import flatdim
import src.network.torch_utils as torch_utils
from matplotlib.ticker import StrMethodFormatter

class Evaluation:
    def __init__(self, cfg):
        self.env = cfg.train_env
        self.eval_env = cfg.eval_env
        self.observation, info = self.env.reset(seed=cfg.seed)
        self.eval_env.reset(seed=cfg.seed)

        self.state_dim = flatdim(self.env.observation_space)
        self.action_dim = flatdim(self.env.action_space)

        self.gamma = cfg.gamma
        self.seed = cfg.seed
        
        self.logger = cfg.logger
        self.timeout = cfg.timeout
        self.ep_reward = 0
        self.ep_returns = []
        self.ep_step = 0
        self.total_steps = 0
        self.num_episodes = 0
        self.update_freq = cfg.update_freq
        self.stats_queue_size = cfg.stats_queue_size
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.ep_returns_queue_train = np.zeros(self.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(self.stats_queue_size)
        self.evaluation_criteria = cfg.evaluation_criteria
        self.device = cfg.device
        self.log_interval = cfg.log_interval
        
        self.info_log = []
        if cfg.render == 1: # show the plot while running
            plt.ion()
            self.eval_fig = plt.figure(figsize=(12, 4))
            self.eval_ax1 = self.eval_fig.add_subplot(131)
            self.eval_ax1.set_ylim(self.env.visualization_range)
            self.eval_ax2 = self.eval_fig.add_subplot(132)
            self.eval_ax3 = self.eval_fig.add_subplot(133)
            self.eval_line1 = None
            self.eval_line2 = None
            self.eval_line3 = None
            self.render = self.render_online
            self.save_render = self.save_render_online
        elif cfg.render == 2:
            self.eval_line1 = []
            self.eval_line2 = []
            self.eval_line3 = []
            self.render = self.render_offline
            self.save_render = self.save_render_offline

    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter % self.stats_queue_size] = ep_return
        self.train_stats_counter += 1
    
    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter % self.stats_queue_size] = ep_return
        self.test_stats_counter += 1
    
    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.stats_queue_size if total_ep is None else total_ep
        total_steps = 0
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            ep_return, steps, traj = self.eval_episode(log_traj=log_traj)
            total_steps += steps
            total_states += traj[0]
            total_actions += traj[1]
            total_returns += traj[2]
            if self.evaluation_criteria == "return":
                self.add_test_log(ep_return)
                if initialize:
                    self.add_train_log(ep_return)
            elif self.evaluation_criteria == "steps":
                self.add_test_log(steps)
                if initialize:
                    self.add_train_log(steps)
            else:
                raise NotImplementedError
        return [total_states, total_actions, total_returns]
    
    def eval_step(self, state):
        a, _, _ = self.get_policy(torch_utils.tensor(self.state_normalizer(state), self.device), False)
        a = torch_utils.to_np(a)
        return a
    
    def get_policy(self, observation, with_grad):
        raise NotImplementedError
    
    def eval_episode(self, log_traj=False):
        ep_traj = []
        state, _ = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.eval_step(state.reshape((1, -1)))[0]
            last_state = state
            state, reward, done, _, _ = self.eval_env.step(action)
            if log_traj:
                ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.timeout:
                break
        
        states = []
        actions = []
        returns = []
        if log_traj:
            ret = 0
            for i in range(len(ep_traj) - 1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.gamma * ret

                returns.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, returns]
    
    def log_return(self, log_ary, name, elapsed_time):
        rewards = log_ary
        total_episodes = len(self.ep_returns)
        mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
        
        if self.log_interval and not self.total_steps % self.log_interval:
            log_str = '%s LOG: steps %d, episodes %3d, ' \
                      'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'
            
            self.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                    min_, max_, len(rewards), elapsed_time))
        
        return mean, median, min_, max_
    
    def log_file(self, elapsed_time=-1, test=True):
        train_mean, train_median, train_min_, train_max_ = self.log_return(self.ep_returns_queue_train[: min(self.train_stats_counter, self.stats_queue_size)],
                                                                           "TRAIN", elapsed_time)
        # try:
        #     normalized = np.array([self.env.env.unwrapped.get_normalized_score(ret_) for ret_ in self.ep_returns_queue_train])
        #     train_mean, train_median, train_min_, train_max_ = self.log_return(normalized, "TRAIN Normalized", elapsed_time)
        # except:
        #     pass
        
        if test:
            self.populate_states, self.populate_actions, self.populate_sampled_returns = self.populate_returns(log_traj=True)
            self.populate_latest = True
            test_mean, test_median, test_min_, test_max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            # try:
            #     normalized = np.array([self.eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in self.ep_returns_queue_test])
            #     test_mean, test_median, test_min_, test_max_ = self.log_return(normalized, "TEST Normalized", elapsed_time)
            # except:
            #     pass
        else:
            test_mean, test_median, test_min_, test_max_ = [np.nan] * 4
        return train_mean, train_median, train_min_, train_max_, test_mean, test_median, test_min_, test_max_
    
    def render_online(self, ary1, ary2_info, ary3):
        ary2, ary2_coord = ary2_info
        if len(ary1)==0:
            return
        if self.eval_line1 is None:
            self.eval_line1, = self.eval_ax1.plot(ary1, 'r-')
            self.eval_line2 = self.eval_ax2.imshow(ary2)
            plt.colorbar(self.eval_line2, ax=self.eval_ax2)
            self.eval_ax2.set_xticks(np.arange(ary2_coord.shape[2]),
                                     labels=["{:.2f}".format(x) for x in ary2_coord[0, 0, :]], rotation=90)
            self.eval_ax2.set_xlabel("dim 0")
            self.eval_ax2.set_yticks(np.arange(ary2_coord.shape[1]),
                                     labels=["{:.2f}".format(x) for x in ary2_coord[1, :, 0]])
            self.eval_ax2.set_ylabel("dim 1")

            self.eval_line3 = self.eval_ax3.imshow(ary3['sum'])
            plt.colorbar(self.eval_line3, ax=self.eval_ax3)
            ary3_shape = len(ary3['sum'])
            ary3_labels = np.linspace(0.0, 1.0, ary3_shape + 1)
            self.eval_ax3.set_xticks(np.arange(ary3_shape + 1) - 0.5, labels=["{:.2f}".format(x) for x in ary3_labels], rotation=90)
            self.eval_ax3.set_xlabel("dim 0")
            self.eval_ax3.set_yticks(np.arange(ary3_shape + 1) - 0.5, labels=["{:.2f}".format(x) for x in ary3_labels])
            self.eval_ax3.set_ylabel("dim 1")

            dot = self.eval_ax3.scatter(ary3['curr_action'][0], ary3['curr_action'][1], color = 'red')

            self.eval_fig.tight_layout()
            plt.show()
            dot.remove()
        else:
            self.eval_line1.set_ydata(ary1)
            self.eval_line2.set_array(ary2)
            self.eval_line2.set_clim(vmin=ary2.min(), vmax=ary2.max())
            self.eval_line3.set_array(ary3['sum'])
            self.eval_line3.set_clim(vmin=ary3['sum'].min(), vmax=ary3['sum'].max())
            dot = self.eval_ax3.scatter(ary3['curr_action'][0], ary3['curr_action'][1], color = 'red')

            self.eval_fig.canvas.draw()
            self.eval_fig.canvas.flush_events()
            dot.remove()

    def render_offline(self, ary1, ary2_info, ary3):
        if len(ary1)==0:
            return
        self.eval_line1.append(ary1)
        self.eval_line2.append(ary2_info[0])
        self.ary2_coord = ary2_info[1]
        self.eval_line3.append(ary3)
        # self.ary3_coord = ary3_info[1]

    # def save_frames_as_gif(self, frames, filename):
    #     imageio.mimsave(filename+".gif",  # output gif
    #                     frames,  # array of input frames
    #                     duration=50)

    def save_frames_as_mp4(self, ary1, ary2, ary3, filename, text=None, clip_l=-2, clip_u=1):
        plt.ioff()
        eval_fig = plt.figure(figsize=(12, 4))
        eval_ax1 = eval_fig.add_subplot(131)
        eval_ax2 = eval_fig.add_subplot(132)
        eval_ax3 = eval_fig.add_subplot(133)
    
        writer = imageio.get_writer(filename+".mp4", fps=20)
        eval_line1, = eval_ax1.plot(ary1[0])
        eval_ax1.set_title(0)
        eval_ax1.set_ylim(self.env.visualization_range)
        
        eval_line2 = eval_ax2.imshow(ary2[0])
        plt.colorbar(eval_line2, ax=eval_ax2)
        assert self.ary2_coord.shape[0] == 2 # only works for 2 dimension space
        eval_ax2.set_xticks(np.arange(self.ary2_coord.shape[2]), labels=["{:.2f}".format(x) for x in self.ary2_coord[0, 0, :]], rotation=90)
        eval_ax2.set_xlabel("dim 0")
        eval_ax2.set_yticks(np.arange(self.ary2_coord.shape[1]), labels=["{:.2f}".format(x) for x in self.ary2_coord[1, :, 0]])
        eval_ax2.set_ylabel("dim 1")

        eval_line3 = eval_ax3.imshow(ary3[0]['sum'], vmin=np.array(ary3[0]['sum']).min(), vmax=np.array(ary3[0]['sum']).max())
        plt.colorbar(eval_line3, ax=eval_ax3)
        ary3_shape = len(ary3[0]['sum'])
        ary3_labels = np.linspace(0.0, 1.0, ary3_shape + 1)
        eval_ax3.set_xticks(np.arange(ary3_shape + 1) - 0.5, labels=["{:.2f}".format(x) for x in ary3_labels], rotation=90)
        eval_ax3.set_xlabel("dim 0")
        eval_ax3.set_yticks(np.arange(ary3_shape + 1) - 0.5, labels=["{:.2f}".format(x) for x in ary3_labels])
        eval_ax3.set_ylabel("dim 1")

        eval_fig.tight_layout()
        for idx, [curve, q_heatmap, visit_heatmap] in enumerate(zip(ary1[0:], ary2[0:], ary3[0:])):
            eval_line1.set_ydata(curve)
            eval_line2.set_array(q_heatmap)
            eval_line2.set_clim(vmin=clip_l, vmax=clip_u)
            eval_line3.set_array(visit_heatmap['sum'])
            eval_line3.set_clim(vmin=visit_heatmap['sum'].min(), vmax=visit_heatmap['sum'].max())
            dot = eval_ax3.scatter(visit_heatmap['curr_action'][0], visit_heatmap['curr_action'][1], color = 'red')
            eval_ax1.title.set_text(idx)
            eval_fig.canvas.draw()
            eval_fig.canvas.flush_events()
            
            data = np.frombuffer(eval_fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(eval_fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(data)

            dot.remove()

            # writer.append_data(imageio.v3.imread(im))
        writer.close()

    def save_render_offline(self, vis_dir):
        self.save_frames_as_mp4(self.eval_line1, self.eval_line2, self.eval_line3, os.path.join(vis_dir, "render"))

    def save_render_online(self, vis_dir):
        return
        
    def save_info(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.info_log, f)