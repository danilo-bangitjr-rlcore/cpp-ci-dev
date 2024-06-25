import numpy as np
import pandas as pd
import gymnasium

from datetime import timedelta

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.data_loaders.base import BaseDataLoader
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data import TransitionBatch, Transition


class OfflineAnytimeInteraction(AnytimeInteraction):
    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert,
            data_loader: BaseDataLoader
    ):
        super().__init__(cfg, env, state_constructor, alerts)
        self.obs_duration = timedelta(seconds=self.obs_length)

        self.data_loader = data_loader
        self.df = self.data_loader.load_data(
            self.data_loader.test_filenames)  # Assuming df only has actions of duration self.obs_length * self.steps_per_decision

        assert not self.df.isnull().values.any()
        assert np.isnan(self.df.to_numpy()).any() == False

        self.action_df = self.df[self.data_loader.action_col_names]

        self.obs_df = self.df[self.data_loader.obs_col_names]
        self.curr_time = self.df.iloc[0].name  # pd.Timestamp
        self.end_time = self.df.iloc[-1].name  # pd.Timestamp

        self.signal_col_names = cfg.signal_col_names
        self.signal_inds = cfg.signal_inds

    def warmup_sc(self):
        """
        Warm up state constructor. Assuming no discontinuities in df
        """
        start_ind = self.obs_df.iloc[0].name
        warmup_sec = timedelta(seconds=self.obs_length * self.warmup_steps)
        elapsed_warmup = timedelta(seconds=0)
        action_start = start_ind
        warmup_end = start_ind + warmup_sec

        self.state_constructor.reset()
        initial_state = True

        while elapsed_warmup < warmup_sec:
            self.curr_action, self.action_end, next_action_start, _, _, _ = self.data_loader.find_action_boundary(
                self.action_df, action_start)
            self.norm_curr_action = self.action_normalizer(self.curr_action)

            curr_action_steps, step_start = self.data_loader.get_curr_action_steps(action_start, self.action_end)
            step_remainder = curr_action_steps % self.steps_per_decision
            self.steps_since_decision = ((self.steps_per_decision - step_remainder) + 1) % self.steps_per_decision

            for i in range(curr_action_steps):
                decision_point = self.steps_since_decision == 0

                step_end = step_start + timedelta(seconds=self.obs_length)
                raw_obs = self.data_loader.get_obs(self.obs_df, step_start, step_end)
                self.obs = self.obs_normalizer(raw_obs)

                self.state = self.state_constructor(self.obs,
                                                    self.norm_curr_action,
                                                    initial_state=initial_state,
                                                    decision_point=decision_point,
                                                    steps_since_decision=self.steps_since_decision)

                initial_state = False
                self.steps_since_decision = (self.steps_since_decision + 1) % self.steps_per_decision
                step_start += timedelta(seconds=self.obs_length)
                elapsed_warmup += timedelta(seconds=self.obs_length)
                if elapsed_warmup >= warmup_sec:
                    warmup_end = step_start
                    break

            action_start = next_action_start

        self.curr_time = warmup_end
        self.prev_decision_point = decision_point

    def step(self) -> tuple[list[Transition], list[Transition], list[Transition], list[dict], list[dict]]:
        """
        Process the offline data, as if it were encountered online, until the next decision point is reached.
        Create transitions using the 'Anytime' paradigm
        Returns:
        - new_agent_transitions: List of all produced agent transitions
        - agent_train_transitions: List of Agent transitions that didn't trigger an alert
        - alert_train_transitions: List of Alert transitions that didn't trigger an alert
        - alert_info_list: List of dictionaries describing which types of alerts were/weren't triggered
        - env_info_list: List of dictionaries describing env info
        """
        partial_transitions = []  # contains (O, S, A, R, S_DP, O', S', C)
        alert_info_list = []
        trunc = False

        # If at the end of an action window, find the next action window in the DF and align the time steps
        if self.curr_time >= self.action_end:
            self.curr_action, self.action_end, _, _, _, _ = self.data_loader.find_action_boundary(self.action_df,
                                                                                                  self.curr_time)
            self.norm_curr_action = self.action_normalizer(self.curr_action)

            curr_action_steps, step_start = self.data_loader.get_curr_action_steps(self.curr_time, self.action_end)
            step_remainder = curr_action_steps % self.steps_per_decision
            self.steps_since_decision = ((self.steps_per_decision - step_remainder) + 1) % self.steps_per_decision

        # Iterate until the next decision point and create transitions for the observed states
        for i in range(self.steps_since_decision, self.steps_per_decision + 1, 1):
            if self.curr_time >= self.end_time:
                # Reached end of offline data eval DF
                trunc = True
                break
            else:
                decision_point = self.steps_since_decision == 0

                step_end = self.curr_time + timedelta(seconds=self.obs_length)

                # NOTE: this may return nan if there is a gap in the df
                raw_next_obs = self.data_loader.get_obs(self.obs_df, self.curr_time, step_end)

                next_obs = self.obs_normalizer(raw_next_obs)

                next_state = self.state_constructor(next_obs,
                                                    self.norm_curr_action,
                                                    initial_state=False,
                                                    decision_point=decision_point,
                                                    steps_since_decision=self.steps_since_decision)

                raw_reward = self.env._get_reward(raw_next_obs, self.curr_action)
                reward = self.reward_normalizer(raw_reward)

                curr_cumulants = self.get_cumulants(reward, next_obs)

                alert_info = {}
                alert_info["raw_action"] = [self.curr_action]
                alert_info["action"] = [self.norm_curr_action]
                alert_info["state"] = [self.state]
                alert_info["reward"] = [reward]
                alert_info["next_obs"] = [next_obs]

                # Additional signals (sensors) we want to plot
                for j in range(len(self.signal_col_names)):
                    alert_info[self.signal_col_names[j]] = [raw_next_obs[self.signal_inds[j]]]

                step_alert_info = self.alerts.evaluate(**alert_info)
                for key in step_alert_info:
                    alert_info[key] = step_alert_info[key]

                alert_info_list.append(alert_info)

                partial_transitions.append((self.obs,
                                            self.state,
                                            self.norm_curr_action,
                                            reward,
                                            self.prev_decision_point,
                                            next_obs,
                                            next_state,
                                            curr_cumulants))

                self.curr_time += self.obs_duration
                self.prev_decision_point = decision_point
                self.state = next_state
                self.obs = next_obs
                self.steps_since_decision = (self.steps_since_decision + 1) % self.steps_per_decision

        self.steps_since_decision = 1
        new_agent_transitions, new_alert_transitions = self.create_n_step_transitions(partial_transitions,
                                                                                      self.state,
                                                                                      self.obs,
                                                                                      False,  # Assuming continuing env
                                                                                      trunc)

        # Only train on transitions where there weren't any alerts
        agent_train_transitions = self.get_agent_train_transitions(new_agent_transitions, alert_info_list)
        alert_train_transitions = self.get_alert_train_transitions(new_alert_transitions, alert_info_list)

        return (new_agent_transitions, agent_train_transitions, alert_train_transitions, alert_info_list, [{}])

