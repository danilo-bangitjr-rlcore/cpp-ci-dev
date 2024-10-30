import torch
import torch.nn as nn
import numpy as np
import random
import logging

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Optional

import corerl.calibration_models.utils as utils
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.calibration_models.base import BaseCalibrationModel
from corerl.data.data import OldObsTransition
from corerl.component.network.utils import tensor, to_np
from corerl.data.data import Trajectory
from corerl.agent.base import BaseAgent

log = logging.getLogger(__name__)


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
        self.interpolation = cfg.interpolation

    def _eval(self, batch, with_grad):
        # gamma_exponents double as the durations of actions
        duration = batch.gamma_exponent
        endo_next_obs_batch = batch.boot_obs[:, self.endo_inds]
        duration /= self.max_action_duration
        prediction = self.get_prediction(batch.state, batch.action, duration, with_grad=with_grad)
        loss = nn.functional.mse_loss(prediction, endo_next_obs_batch)

        return loss

    def _update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)[0]
        loss = self._eval(batch, with_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(closure=lambda: 0.)

        self.train_losses.append(loss.detach().numpy())

    def train(self):
        log.info('Training model...')
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
                test_batch = self.test_buffer.sample_mini_batch(self.batch_size)[0]
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

        # Revan: I would like to unify this with _do_rollout in base but dk how
        if start_idx is None:
            start_idx = random.randint(0, traj_cm.num_transitions - self.max_rollout_len - 1)

        transitions_cm = traj_cm.transitions[start_idx:]
        sc_cm = traj_cm.get_sc_at_idx(start_idx)  # state constructor for the model
        state_cm = transitions_cm[0].state  # initial state for the model

        state_agent = None  # the state for the agent is not used unless we pass in an agent
        sc_agent = None
        use_agent = False

        if agent is not None:
            assert traj_agent is not None
            transitions_agent = traj_agent.transitions[start_idx:]

            sc_agent = traj_agent.get_sc_at_idx(start_idx)  # state constructor for the agent
            state_agent = transitions_agent[0].state
            use_agent = True

        g = 0  # the return

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions_cm), self.max_rollout_len)

        steps_until_decision = None
        action = transitions_cm[0].action  # the initial agent's action
        decision_point = transitions_cm[0].state_dp

        # we need the following variables for constructing observation transitions
        prev_action = None
        prev_obs = transitions_cm[0].obs
        prev_steps_until_decision = None
        prev_decision_point = decision_point

        # these lists are used to construct transitions.
        curr_decision_obs_transitions = []
        curr_decision_states = []
        if use_agent:
            # initialize this list with the first state that the agent sees
            curr_decision_states.append(state_agent)

        outer_step = 0
        done = False
        num_predictions = 0
        curr_obs = transitions_cm[0].obs

        while not done:
            transition_step = transitions_cm[outer_step]
            if steps_until_decision is None or not use_agent:
                steps_until_decision = transition_step.gamma_exponent
            elif steps_until_decision == 0:
                steps_until_decision = self.steps_per_decision

            if prev_steps_until_decision is None:
                prev_steps_until_decision = steps_until_decision

            action = self._get_action(action, transition_step, decision_point,
                                      use_agent=use_agent, agent=agent, state_agent=state_agent)

            duration_int = steps_until_decision  # how long to hold this action for
            duration = steps_until_decision / self.max_action_duration

            state_cm_tensor = tensor(state_cm).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))
            duration_tensor = tensor(duration).reshape((1, -1))

            fictitious_next_obs = None

            predicted_next_endo_obs = to_np(self.get_prediction(state_cm_tensor, action_tensor, duration_tensor))
            for inner_step in range(1, duration_int+1):
                inner_step_transition = transitions_cm[outer_step + inner_step]
                inner_step_obs = inner_step_transition.obs
                inner_next_obs = inner_step_transition.next_obs

                assert inner_next_obs is not None

                if self.interpolation == 'linear':
                    predicted_inner_endo_obs = self.linear_interpolation(inner_step, duration_int, curr_obs,
                                                                         predicted_next_endo_obs)
                else:  # use the model to interpolate
                    inner_step_duration = inner_step / self.max_action_duration
                    inner_duration_tensor = tensor(inner_step_duration).reshape((1, -1))
                    predicted_inner_endo_obs = to_np(
                        self.get_prediction(state_cm_tensor, action_tensor, inner_duration_tensor))

                loss_step = np.mean(np.abs(inner_next_obs[self.endo_inds] - to_np(predicted_inner_endo_obs)))
                losses.append(loss_step)

                fictitious_next_obs = utils.new_fictitious_obs(predicted_inner_endo_obs, inner_next_obs, self.endo_inds)

                # update the state constructors
                steps_until_decision -= 1
                decision_point = steps_until_decision == 0

                state_cm = sc_cm(fictitious_next_obs, action,
                                 decision_point=decision_point,
                                 steps_until_decision=steps_until_decision)

                r = self._get_reward(prev_action, action, fictitious_next_obs)
                g += self.gamma ** (outer_step + inner_step) * r

                if use_agent and sc_agent is not None:
                    state_agent = sc_agent(fictitious_next_obs, action,
                                           decision_point=decision_point,
                                           steps_until_decision=steps_until_decision)

                    obs_transition = OldObsTransition(
                        prev_action,
                        prev_obs,
                        prev_steps_until_decision,
                        prev_decision_point,
                        action,
                        r,
                        fictitious_next_obs,
                        steps_until_decision,
                        decision_point,
                        False,  # termination false
                        False,  # truncation false
                        gap=False)  # assume no data gap

                    curr_decision_obs_transitions.append(obs_transition)
                    curr_decision_states.append(state_agent)

                    curr_decision_obs_transitions, curr_decision_states = self._make_transitions_and_update(agent,
                                                                                                            decision_point,
                                                                                                            curr_decision_obs_transitions,
                                                                                                            curr_decision_states)

                prev_action = action
                prev_obs = fictitious_next_obs
                prev_steps_until_decision = steps_until_decision
                prev_decision_point = decision_point

                assert inner_step_obs is not None

                actions.append(action)
                endo_obss.append(inner_step_obs[0])
                predicted_endo_obss.append(predicted_inner_endo_obs)

                if outer_step + inner_step > rollout_len:
                    done = True
                    break

            outer_step += duration_int
            num_predictions += 1

            # log the loss
            curr_obs = fictitious_next_obs

        if plot is not None:
            self._plot(endo_obss, actions, predicted_endo_obss, plot_save_path, plot, start_idx)

        return g, losses
