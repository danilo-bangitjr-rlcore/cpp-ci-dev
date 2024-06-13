import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from corerl.component.buffer.factory import init_buffer

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer

from corerl.component.network.utils import tensor, to_np
import matplotlib.pyplot as plt


class AnytimeCalibrationModel:
    def __init__(self, cfg: DictConfig, train_info):
        self.test_trajectories = train_info['test_trajectories']
        train_transitions = train_info['train_transitions']
        self.reward_func = train_info['reward_func']
        self.interaction = train_info['interaction']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)

        test_transitions = []
        for t in self.test_trajectories:
            test_transitions += t.transitions

        self.test_buffer.load(test_transitions)

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        input_dim = len(train_transitions[0].state)
        action_dim = len(train_transitions[0].action)
        output_dim = len(train_transitions[0].obs[self.endo_inds])

        # the plus one is for the duration until the next observation
        self.model = init_custom_network(cfg.model, input_dim=input_dim + action_dim + 1, output_dim=output_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.max_action_duration = 30

    def eval(self, batch, with_grad):
        # gamma_exponents double as the durations of actions
        state_batch, action_batch, next_obs_batch, duration = batch.state, batch.action, batch.boot_obs, batch.gamma_exponent
        endo_next_obs_batch = next_obs_batch[:, self.endo_inds]

        duration /= self.max_action_duration
        prediction = self.get_prediction(state_batch, action_batch, duration, with_grad=with_grad)
        loss = nn.functional.mse_loss(prediction, endo_next_obs_batch)

        return loss

    def update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)
        loss = self.eval(batch, with_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def train(self):
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        train_losses = []
        train_loss = 0
        test_loss = 0
        window_avg = 100
        test_losses = []
        for itr in pbar:
            self.update()

            if len(self.train_losses) >= window_avg:
                train_loss = np.mean(self.train_losses[-window_avg:])

                train_losses.append(train_loss)

            if itr % 100 == 0:
                test_batch = self.test_buffer.sample_mini_batch(self.batch_size)
                test_loss = self.eval(test_batch, with_grad=False).detach().numpy()
                test_losses.append(test_loss)

            pbar.set_description("train loss: {:7.6f}, test_loss: {:7.6f}".format(train_loss, test_loss))

        self.do_test_rollouts()
        # plt.plot(train_losses)
        # plt.yscale('log')
        #
        # plt.plot(test_losses)
        #
        # plt.show()

        return self.train_losses, self.test_losses

    def do_test_rollout(self, traj, num, start):
        # validates the model's accuracy on a test rollout
        transitions = traj.transitions[start:]
        sc = deepcopy(traj.scs[start])
        losses = []
        rollout_len = min(traj.num_transitions, self.max_rollout_len)

        state = transitions[0].state

        orps = []
        predicted_orps = []
        actions = []

        step = 0
        done = False

        num_predictions = 0
        curr_obs = transitions[0].obs
        while not done:
            t = transitions[step]
            action = t.action

            boot_obs = t.boot_obs
            boot_endo_obs = boot_obs[self.endo_inds]
            duration_int = t.gamma_exponent
            duration = duration_int / self.max_action_duration

            state_tensor = tensor(state).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))
            duration_tensor = tensor(duration).reshape((1, -1))

            # generate a sequence of fictitious observations

            fictitious_obss = []
            print("duration_int", duration_int)
            # I'm not sure the for loop here is entirely correct.

            predicted_next_endo_obs = to_np(self.get_prediction(state_tensor, action_tensor, duration_tensor))
            # for inter_step in range(1, duration_int + 1):
            #     print("inter_step", inter_step)
            #     inter_step_transition = transitions[step + inter_step]
            #     inter_step_obs = inter_step_transition.obs
            #     print("action", inter_step_transition.action)
            #
            #     inter_step_duration = inter_step / self.max_action_duration
            #     inter_duration_tensor = tensor(inter_step_duration).reshape((1, -1))
            #     predicted_inter_endo_obs = to_np(
            #         self.get_prediction(state_tensor, action_tensor, inter_duration_tensor))
            #     if inter_step == duration_int:
            #         print('prediction_1', predicted_inter_endo_obs)
            #         print('prediction_2', predicted_next_endo_obs)
            #
            #     fictitious_obs = inter_step_obs.copy()
            #     for i, j in enumerate(self.endo_inds):
            #         fictitious_obs[j] = predicted_inter_endo_obs[i]
            #
            #     # assert transitions[step + inter_step].action == action
            #     decision_point = step + inter_step % self.steps_per_decision == 0
            #
            #     state = sc(fictitious_obs, action, decision_point=decision_point)
            #
            #     loss_step = np.mean(np.abs(inter_step_obs[self.endo_inds] - to_np(predicted_inter_endo_obs)))
            #     losses.append(loss_step)
            #
            #     actions.append(action)
            #     orps.append(inter_step_obs[0])
            #     predicted_orps.append(predicted_inter_endo_obs)

            for inter_step in range(1, duration_int + 1):
                w = (inter_step) / duration_int
                fictitious_endo_obs = (1 - w) * curr_obs[self.endo_inds] + w * predicted_next_endo_obs

                inter_step_obs = transitions[step + inter_step].obs
                fictitious_obs = inter_step_obs.copy()
                for i, j in enumerate(self.endo_inds):
                    fictitious_obs[j] = fictitious_endo_obs[i]

                # assert transitions[step + inter_step].action == action
                decision_point = step + inter_step % self.steps_per_decision == 0
                state = sc(fictitious_obs, action, decision_point=decision_point)

                actions.append(action)
                orps.append(inter_step_obs[0])
                predicted_orps.append(fictitious_endo_obs)

            step += duration_int
            num_predictions += 1

            # log the loss

            curr_obs = fictitious_obs

            if step > rollout_len:
                done = True

        print('NUM PREDICTIONS', num_predictions)

        plt.plot(orps, label='orps')
        plt.plot(actions, label='actions')

        predicted_orps = [np.squeeze(to_np(p)) for p in predicted_orps]
        plt.plot(predicted_orps, label='predicted orps')
        plt.legend()

        plt.xlabel("Rollout Step")
        plt.savefig(f"test_{num}_{start}.png", bbox_inches='tight')
        plt.clf()

        return losses

    def do_test_rollouts(self):
        import matplotlib.pyplot as plt
        import random
        for n, test_traj in enumerate(self.test_trajectories):
            last = test_traj.num_transitions - self.max_rollout_len
            num_rollouts = 20
            increase_idx = last // num_rollouts
            start_idx = 0
            for start in range(num_rollouts):
                # start_idx = random.choice(range(0, test_traj.num_transitions-self.max_rollout_len))
                losses = self.do_test_rollout(test_traj, n,  start_idx)
                start_idx += increase_idx

        #         plt.plot(np.array(losses), c='b', alpha=0.2)  # * (471.20947 - 2.6265918e+02)
        # plt.ylabel("Absolute Error From True ORP")
        # plt.xlabel("Rollout Step")
        # plt.show()

    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, duration: torch.Tensor,
                       with_grad: bool = False):
        x = torch.concat((state, action, duration), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def do_agent_rollout(self, traj, agent, rollout_len=20):
        pass
