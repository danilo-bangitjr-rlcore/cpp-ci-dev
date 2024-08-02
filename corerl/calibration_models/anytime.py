import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from typing import Optional

import corerl.calibration_models.utils as utils
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.calibration_models.base import BaseCalibrationModel
from corerl.data.data import Transition, ObsTransition
from corerl.component.network.utils import tensor, to_np
from corerl.data.data import Trajectory
from corerl.agent.base import BaseAgent


class AnytimeCalibrationModel(BaseCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info):
        super().__init__(cfg, train_info)
        train_transitions = train_info['train_transitions_cm']
        test_transitions = train_info['test_transitions_cm']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)
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

        # this is for duration normalization. Could be different in the future.
        self.max_action_duration = cfg.steps_per_decision
        self.interpolation = 'not'

    def _eval(self, batch, with_grad):
        # gamma_exponents double as the durations of actions
        state_batch, action_batch, next_obs_batch, duration = batch.state, batch.action, batch.boot_obs, batch.gamma_exponent
        endo_next_obs_batch = next_obs_batch[:, self.endo_inds]
        duration /= self.max_action_duration
        prediction = self.get_prediction(state_batch, action_batch, duration, with_grad=with_grad)
        loss = nn.functional.mse_loss(prediction, endo_next_obs_batch)

        return loss

    def _update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)
        loss = self._eval(batch, with_grad=True)
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
            self._update()
            if len(self.train_losses) >= window_avg:
                train_loss = np.mean(self.train_losses[-window_avg:])
                train_losses.append(train_loss)

            if itr % 100 == 0:
                test_batch = self.test_buffer.sample_mini_batch(self.batch_size)
                test_loss = self._eval(test_batch, with_grad=False).detach().numpy()
                test_losses.append(test_loss)

            pbar.set_description("train loss: {:7.6f}, test_loss: {:7.6f}".format(train_loss, test_loss))

        return self.train_losses, self.test_losses

    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, duration: torch.Tensor,
                       with_grad: bool = False):
        x = torch.concat((state, action, duration), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def _get_next_endo_obs(self, state, action, kwargs):
        pass

    def linear_interpolation(self, inter_step, duration_int, curr_obs, predicted_next_endo_obs):
        w = inter_step / duration_int
        fictitious_endo_obs = (1 - w) * curr_obs[self.endo_inds] + w * predicted_next_endo_obs
        return fictitious_endo_obs

    def _do_rollout(self,
                    traj_cm: Trajectory,
                    agent: Optional[BaseAgent] = None,
                    traj_agent: Optional[Trajectory] = None,
                    start_idx: Optional[int] = None,
                    plot=None,
                    plot_save_path=None,
                    ) -> tuple[float, list[float]]:

        # I would like to unify this with _do
        if start_idx is None:
            start_idx = random.randint(0, traj_cm.num_transitions - self.max_rollout_len - 1)
        transitions_cm = traj_cm.transitions[start_idx:]
        sc_cm = deepcopy(traj_cm.scs[start_idx])  # state constructor for the model
        state_cm = transitions_cm[0].state  # initial state for the model
        state_agent = None
        sc_agent = None
        use_agent = False

        if agent is not None:
            assert traj_agent is not None
            transitions_agent = traj_agent.transitions[start_idx:]
            for i in range(len(transitions_cm)):
                assert np.array_equal(transitions_cm[i].obs, transitions_agent[i].obs)
                assert np.array_equal(transitions_cm[i].action, transitions_agent[i].action)
                assert np.array_equal(transitions_cm[i].next_obs, transitions_agent[i].next_obs)

            sc_agent = deepcopy(traj_agent.scs[start_idx])  # state constructor for the agent
            state_agent = transitions_agent[0].state
            use_agent = True

        g = 0  # the return

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions_cm), self.max_rollout_len)

        steps_until_decision_point = None
        steps_since_decision_point = transitions_cm[0].steps_since_decision  # how long since the last decision
        action = transitions_cm[0].action  # the initial agent's action
        decision_point = transitions_cm[0].state_dp

        # we need the following variables for constructing observation transitions
        prev_action = None
        prev_obs = transitions_cm[0].obs
        prev_steps_since_decision_point = steps_since_decision_point
        prev_decision_point = decision_point

        if use_agent:
            # these lists are used to construct transitions.
            curr_decision_obs_transitions = []
            curr_decision_states = [state_agent]  # initialize this list with the first state that the agent sees

        outer_step = 0
        done = False
        num_predictions = 0
        curr_obs = transitions_cm[0].obs

        while not done:
            transition_step = transitions_cm[outer_step]
            if steps_until_decision_point is None or not use_agent:
                steps_until_decision_point = transition_step.gamma_exponent
            elif steps_until_decision_point == 0:
                steps_until_decision_point = self.steps_per_decision

            action = self._get_action(action, transition_step, decision_point,
                                      use_agent=use_agent, agent=agent, state_agent=state_agent)

            duration_int = steps_until_decision_point  # how long to hold this action for
            duration = steps_until_decision_point / self.max_action_duration

            state_cm_tensor = tensor(state_cm).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))
            duration_tensor = tensor(duration).reshape((1, -1))

            predicted_next_endo_obs = to_np(self.get_prediction(state_cm_tensor, action_tensor, duration_tensor))
            for inter_step in range(0, duration_int):
                inter_step_transition = transitions_cm[outer_step + inter_step]
                inter_step_obs = inter_step_transition.obs
                inter_next_obs = inter_step_transition.next_obs

                if self.interpolation == 'linear':
                    predicted_inter_endo_obs = self.linear_interpolation(inter_step, duration_int, curr_obs,
                                                                         predicted_next_endo_obs)
                else:  # use the model to interpolate
                    inter_step_duration = inter_step / self.max_action_duration
                    inter_duration_tensor = tensor(inter_step_duration).reshape((1, -1))
                    predicted_inter_endo_obs = to_np(
                        self.get_prediction(state_cm_tensor, action_tensor, inter_duration_tensor))

                loss_step = np.mean(np.abs(inter_next_obs[self.endo_inds] - to_np(predicted_inter_endo_obs)))
                losses.append(loss_step)

                fictitious_next_obs = utils.new_fictitious_obs(predicted_inter_endo_obs, inter_next_obs, self.endo_inds)

                # update the state constructors
                steps_until_decision_point -= 1
                steps_since_decision_point += 1
                decision_point = steps_until_decision_point == 0

                if decision_point:
                    steps_since_decision_point = 0

                state_cm = sc_cm(fictitious_next_obs, action,
                                 decision_point=decision_point,
                                 steps_since_decision=steps_since_decision_point)

                r = self._get_reward(prev_action, action, fictitious_next_obs)
                g += self.gamma ** (outer_step + inter_step) * r

                if use_agent:
                    state_agent = sc_agent(fictitious_next_obs, action,
                                           decision_point=decision_point,
                                           steps_since_decision=steps_since_decision_point)

                    obs_transition = ObsTransition(
                        prev_action,
                        prev_obs,
                        prev_steps_since_decision_point,  # I don't think this variable is actually used
                        prev_decision_point,
                        action,
                        r,
                        fictitious_next_obs,
                        steps_since_decision_point,
                        decision_point,
                        False,  # termination false
                        False,  # truncation false
                        gap=False)  # assume no data gap

                    curr_decision_obs_transitions.append(obs_transition)
                    curr_decision_states.append(state_agent)

                    if decision_point:
                        _, _, agent_transitions = self.transition_creator.make_decision_window_transitions(
                            curr_decision_obs_transitions, curr_decision_states)
                        curr_decision_obs_transitions = []
                        curr_decision_states = [state_agent]

                        for transition in agent_transitions:
                            agent.update_buffer(transition)

                    agent.update()

                prev_action = action
                prev_obs = fictitious_next_obs
                prev_steps_since_decision_point = steps_since_decision_point
                prev_decision_point = decision_point

                actions.append(action)
                endo_obss.append(inter_step_obs[0])
                predicted_endo_obss.append(predicted_inter_endo_obs)

                if outer_step + inter_step > rollout_len:
                    done = True
                    break

            outer_step += duration_int
            num_predictions += 1

            # log the loss
            curr_obs = fictitious_next_obs

        if plot is not None:
            plt.plot(endo_obss, label='endo obs.')
            plt.plot(actions, label='actions')

            predicted_endo_obss = [np.squeeze(to_np(p)) for p in predicted_endo_obss]
            plt.plot(predicted_endo_obss, label='predicted endo obs.')
            plt.legend()

            plt.xlabel("Rollout Step")
            plt.savefig(plot_save_path / f"rollout_{plot}_{start_idx}.png", bbox_inches='tight')
            plt.clf()

        return g, losses
