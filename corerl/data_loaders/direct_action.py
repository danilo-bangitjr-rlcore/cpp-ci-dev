from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm

from datetime import timedelta
import pandas as pd
import numpy as np
import pickle as pkl
import random
from collections import deque
from tqdm import tqdm
from corerl.environment.reward.base import BaseReward
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.data_loaders.base import BaseDataLoader
from copy import deepcopy

class DirectActionDataLoader(BaseDataLoader):
    def __init__(self, cfg: DictConfig):
        self.offline_data_path = Path(cfg.offline_data_path)
        # You can either load all the csvs in the directory or a subset
        if cfg.offline_filenames == []:
            self.offline_filenames = list(self.offline_data_path.glob('*.csv'))
        else:
            self.offline_filenames = [self.offline_data_path / file for file in cfg.offline_filenames]
        self.skip_rows = cfg.skip_rows
        self.header = cfg.header
        self.df_col_names = OmegaConf.to_object(cfg.df_col_names)
        self.obs_col_names = OmegaConf.to_object(cfg.obs_col_names)
        self.control_col_names = OmegaConf.to_object(cfg.control_col_names)
        self.date_col_name = cfg.date_col_name
        self.max_time_delta = cfg.max_time_delta
        self.time_thresh = pd.Timedelta(self.max_time_delta, "s")
        self.steps_per_decision = cfg.steps_per_decision
        self.obs_length = cfg.obs_length
        self.warmup_steps = cfg.warmup_steps
        self.gamma = cfg.gamma
        self.train_split = cfg.train_split
        self.n_step = cfg.n_step
        # if we return the state constructors's internal state for each transition
        self.return_sc_state = cfg.return_sc_state

    def load_data(self) -> pd.DataFrame:
        """
        Read csvs into a single concatenated df sorted by date, containing only the columns in the observation space
        """
        dfs = []
        for file in self.offline_filenames:
            df = pd.read_csv(file, dtype=np.float32, skiprows=self.skip_rows, header=self.header, names=self.df_col_names, index_col=self.date_col_name, parse_dates=True)
            dfs.append(df)
        concat_df = pd.concat(dfs)
        concat_df.sort_values(by=[self.date_col_name], inplace=True)
        concat_df = concat_df[self.obs_col_names]

        print("Concatenated DF:")
        print(concat_df)

        return concat_df

    def get_obs_max_min(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Find the max and min values for each column in the input df to later be used for normalization
        """
        np_min_max = df.agg(['min', 'max']).to_numpy()
        obs_space_low = np_min_max[0, :]
        obs_space_high = np_min_max[1, :]

        print("Obs Space Low:", obs_space_low)
        print("Obs Space High:", obs_space_high)

        return obs_space_low, obs_space_high

    def find_uninterrupted_windows(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        There may be large gaps in time in the input dataframe so we split it up into a list of dataframes that are each uninterrupted
        """
        df_list = []
        time_thresh = pd.Timedelta(self.max_time_delta, "s")
        df_start = df.iloc[0].name
        prev_date = df_start
        for index, row in df.iterrows():
            curr_date = index
            date_diff = curr_date - prev_date
            if date_diff > time_thresh:
                curr_df = self.get_df_date_range(df, df_start, prev_date)
                df_list.append(curr_df)
                df_start = curr_date
            prev_date = curr_date
        final_df = self.get_df_date_range(df, df_start, index)
        df_list.append(final_df)

        print("DF List:")
        print(df_list)

        return df_list

    def get_df_date_range(self, df: pd.DataFrame, start_ind: pd.Timestamp, end_ind: pd.Timestamp) -> pd.DataFrame:
        window_df = df.loc[start_ind: end_ind]

        return window_df

    def warmup_sc(self,
                  df: pd.DataFrame,
                  state_constructor: BaseStateConstructor,
                  interaction: NormalizerInteraction,
                  start_ind: pd.Timestamp) -> (
            BaseStateConstructor, pd.Timestamp, np.ndarray, bool, BaseStateConstructor):
        """
        Warm up state constructor
        """
        warmup_sec = timedelta(seconds=self.obs_length * self.warmup_steps)
        elapsed_warmup = timedelta(seconds=0)
        action_start = start_ind
        warmup_end = start_ind + warmup_sec

        state_constructor.reset()

        while elapsed_warmup < warmup_sec:
            curr_action, action_end, next_action_start, _, _, data_gap = self.find_action_boundary(df, action_start)

            if data_gap:
                elapsed_warmup = timedelta(seconds=0)
                state_constructor.reset()
            else:
                curr_action_steps, step_start = self.get_curr_action_steps(action_start, action_end)
                step_remainder = curr_action_steps % self.steps_per_decision
                steps_since_decision = ((self.steps_per_decision - step_remainder) + 1) % self.steps_per_decision
                for i in range(curr_action_steps):
                    decision_point = steps_since_decision == 0
                    state, sc_at_state, obs = self.get_state(state_constructor,
                                                             interaction,
                                                             df,
                                                             step_start,
                                                             decision_point,
                                                             steps_since_decision)
                    steps_since_decision = (steps_since_decision + 1) % self.steps_per_decision
                    step_start += timedelta(seconds=self.obs_length)
                    elapsed_warmup += timedelta(seconds=self.obs_length)
                    if elapsed_warmup >= warmup_sec:
                        warmup_end = step_start
                        break

            action_start = next_action_start

        return state_constructor, warmup_end, state, decision_point, sc_at_state

    def get_state(self,
                  state_constructor: BaseStateConstructor,
                  interaction: NormalizerInteraction,
                  df: pd.DataFrame,
                  start: pd.Timestamp,
                  decision_point: bool,
                  steps_since_decision: int) -> (np.ndarray, BaseStateConstructor, np.ndarray):

        end = start + timedelta(seconds=self.obs_length)
        obs_df = self.get_df_date_range(df, start, end)
        obs = obs_df.to_numpy()
        obs = interaction.obs_normalizer(obs)
        state = state_constructor(obs, decision_point=decision_point, steps_since_decision=steps_since_decision)
        # copy the state constructor's internal state. This will be used for calibration_models training
        if self.return_sc_state:
            sc_at_state = deepcopy(state_constructor)
        else:
            sc_at_state = None

        return state, sc_at_state, obs

    def find_action_boundary(self, df: pd.DataFrame,
                             start_ind: pd.Timestamp) -> (np.ndarray, pd.Timestamp, pd.Timestamp, bool, bool, bool):
        """
        Return the action taken at the beginning of the dataframe.
        Iterate through the dataframe until an action change, a truncation/termination in the episode, or a large break in time.
        Return the timestamp of the action change/truncation/termination/break in time and whether there was a truncation or termination
        """
        data_gap = False
        prev_date = start_ind
        curr_action = df[self.control_col_names].loc[start_ind].to_numpy()
        for curr_date, row in df.loc[start_ind:].iterrows():
            date_diff = curr_date - prev_date
            # Is there a large gap in time between consecutive rows in the df?
            if date_diff > self.time_thresh:
                data_gap = True
            # Is the episode terminated/truncated?
            trunc, term = self.check_termination_truncation(row)
            row_action = row[self.control_col_names].to_numpy()
            # Has there been a change in action/truncation/termination/gap in data?
            if data_gap or trunc or term or not np.array_equal(curr_action, row_action):
                return curr_action, prev_date, curr_date, trunc, term, data_gap

            prev_date = curr_date

        return curr_action, prev_date, curr_date, False, False, False

    def check_termination_truncation(self, df: pd.DataFrame) -> (bool, bool):
        """
        Project specific. Assuming a continuing environment.
        """
        return False, False

    def create_n_step_transitions_(self,
                                   state_action_rewards: list[tuple],
                                   next_observations: list[np.ndarray],
                                   boot_state: np.ndarray,
                                   term: bool,
                                   trunc: bool) -> (list[tuple], list[tuple]):
        # If n_step = 0, create transitions where all states bootstrap off the state at the next decision point
        # If n_step > 0, create transitions where states bootstrap off the state n steps into the future.
        # If the state n steps ahead is beyond the next decision point, bootstrap off the state at the decision point

        new_transitions = []
        new_obs_transitions = []

        # TODO: rename state_action_rewards
        dp_counter = 1
        if self.n_step == 0:
            n_step_rewards = deque([], self.steps_per_decision)
            boot_state_queue = deque([], self.steps_per_decision)
        else:
            n_step_rewards = deque([], self.n_step)
            boot_state_queue = deque([], self.n_step)

        for i in range(len(state_action_rewards) - 1, -1, -1):
            sar = state_action_rewards[i]
            state = sar[0]
            action = sar[1]
            reward = sar[2]
            s_dp = sar[3] # whether
            boot_state_queue.appendleft(state)

            # Recursively updating n-step reward
            n_step_rewards.appendleft(0.0)
            np_n_step_rewards = np.array(n_step_rewards)
            curr_reward = np.array([reward for i in range(len(n_step_rewards))])
            np_n_step_rewards = curr_reward + (self.gamma * np_n_step_rewards)
            gamma_exp = len(n_step_rewards)
            ns_dp = dp_counter <= boot_state_queue.maxlen

            new_transitions.append((state, action, np_n_step_rewards[-1], boot_state, term, trunc, int(s_dp), int(ns_dp), gamma_exp))
            new_obs_transitions.append((state, action, np_n_step_rewards[-1],  next_observations[i], term, trunc, int(s_dp), int(ns_dp), gamma_exp))

            dp_counter += 1
            n_step_rewards = deque(np_n_step_rewards, n_step_rewards.maxlen)
            if len(boot_state_queue) == boot_state_queue.maxlen: # if queue is full
                boot_state = boot_state_queue[-1]

            if i != len(state_action_rewards) - 1:
                # NOTE: check that next observation is equal to the observation part included in the next state
                # we are not checking the "next_state" in the transition, since that is boot_state, aka
                # the state at the decision point
                assert np.array_equal(state_action_rewards[i + 1][0][0:3], next_observations[i])

        new_transitions.reverse()
        new_obs_transitions.reverse()
        return new_transitions, new_obs_transitions

    def get_curr_action_steps(self, action_start: pd.Timestamp, action_end: pd.Timestamp) -> (int, pd.Timestamp):
        """
        Determine the number of time steps that fit in the given action window and align the time steps, taking into account the time remainder
        """
        action_duration = action_end - action_start
        curr_action_steps = action_duration // timedelta(seconds=self.obs_length)
        time_remainder = action_duration % timedelta(seconds=self.obs_length)
        step_start = action_start + time_remainder

        return curr_action_steps, step_start

    def create_transitions(self,
                           df: pd.DataFrame,
                           state_constructor: BaseStateConstructor,
                           reward_function: BaseReward,
                           interaction: NormalizerInteraction) -> dict:

        """
        Iterate through the df and produce transitions using the "Anytime" paradigm.
        Take into account discontinuities in the dataframe (large gaps in time between consecutive rows)
        Creates fixed n-step transitions or variable n-step transitions that always bootstrap off the state at the next decision point
        """
        train_transitions = []
        test_transitions = []
        train_obs_transitions = []  # transitions where the next state is the next observation, rather than state
        test_obs_transitions = []
        test_scs = []

        # Keep trying to create transitions until you reach the end of the df
        action_start = df.iloc[0].name
        df_end = df.iloc[-1].name
        pbar = tqdm(total=df.index.get_loc(df_end))

        while action_start < df_end:
            data_gap = False # Indicates a discontinuity in the df
            prev_action = None

            # Warmup state constructor using data starting from action_start
            state_constructor, warmup_end, state, warmup_dp, sc_at_state = self.warmup_sc(df,
                                                                                          state_constructor,
                                                                                          interaction,
                                                                                          action_start)

            # Iterate over the action windows in the remainder of the df
            # Produce n-step transitions with the "Anytime" paradigm
            # The 'n' is the number of time steps away from the action boundary
            action_start = warmup_end
            prev_decision_point = warmup_dp
            while not data_gap and action_start < df_end:
                assert len(test_transitions) == len(test_scs) == len(test_obs_transitions)
                curr_action, action_end, next_action_start, trunc, term, data_gap = self.find_action_boundary(df,
                                                                                                              action_start)
                norm_curr_action = interaction.action_normalizer(curr_action)

                # Align time steps within action window
                curr_action_steps, step_start = self.get_curr_action_steps(action_start, action_end)
                step_remainder = curr_action_steps % self.steps_per_decision
                steps_since_decision = ((self.steps_per_decision - step_remainder) + 1) % self.steps_per_decision

                # Iterate over current action time steps and produce (S,A,R)
                state_action_rewards = []
                state_scs = []  # the state constructors at S
                next_observations = []
                for i in range(curr_action_steps):  # how many steps we are talking the current action
                    decision_point = steps_since_decision == 0
                    next_state, sc_at_next_state, next_obs = self.get_state(state_constructor,
                                                                            interaction,
                                                                            df,
                                                                            step_start,
                                                                            decision_point,
                                                                            steps_since_decision)

                    # Any way to make the creation of reward_info more universal?
                    reward_info = {}
                    reward_info['prev_action'] = prev_action
                    reward_info['curr_action'] = curr_action
                    raw_reward = reward_function(interaction.obs_normalizer.denormalize(next_obs), **reward_info)

                    reward = interaction.action_normalizer(raw_reward)
                    state_action_rewards.append((state, norm_curr_action, reward, prev_decision_point))
                    state_scs.append(sc_at_state)  # the state constructor at state
                    next_observations.append(next_obs)  # the next observation for each state

                    # Create n-step Transitions for the states observed since the last decision point
                    if decision_point and len(state_action_rewards) > 0:
                        boot_state = next_state
                        # Set trunc and term to false since we haven't reached the final action boundary within the action window
                        new_transitions, new_obs_transitions = self.create_n_step_transitions_(state_action_rewards,
                                                                                               next_observations,
                                                                                               boot_state,
                                                                                               False,
                                                                                               False)

                        # splits returns a list of (train, test)  for each list passed to self.train_test_split
                        splits = self.train_test_split(new_transitions, new_obs_transitions, state_scs)
                        train_transitions += splits[0][0]
                        test_transitions += splits[0][1]
                        train_obs_transitions += splits[1][0]
                        test_obs_transitions += splits[1][1]
                        test_scs += splits[2][1]

                        # reset lists
                        state_action_rewards = []
                        state_scs = []
                        next_observations = []

                    prev_action = curr_action
                    step_start = step_start + timedelta(seconds=self.obs_length)
                    state = next_state
                    sc_at_state = sc_at_next_state

                    steps_since_decision = (steps_since_decision + 1) % self.steps_per_decision
                    prev_decision_point = decision_point

                    try:
                        pbar.n = df.index.get_loc(step_start)
                        pbar.refresh()
                    except:
                        pass

                action_start = next_action_start

        return_dict = {
            'transitions': (train_transitions, test_transitions),
            'obs_transitions': (train_obs_transitions, test_obs_transitions),
            'test_scs': test_scs
        }

        return return_dict


    def train_test_split(self, *lsts, shuffle: bool = True) -> (list[tuple], list[tuple]):
        num_samples = len(lsts[0])
        for a in lsts:
            assert len(a) == num_samples

        if shuffle:
            lsts = parallel_shuffle(*lsts)

        num_train_samples = int(self.train_split * num_samples)
        train_samples = [lsts[:num_train_samples] for lsts in lsts]
        test_samples = [lsts[num_train_samples:] for lsts in lsts]

        return list(zip(train_samples, test_samples))

    def save(self, save_obj: object, path: Path):
        with open(path, "wb") as file:
            pkl.dump(save_obj, file)

    def load(self, path: Path) -> object:
        with open(path, "rb") as file:
            obj = pkl.load(file)
            return obj


def parallel_shuffle(*args):
    zipped_list = list(zip(*args))
    random.shuffle(zipped_list)
    unzipped = zip(*zipped_list)
    return list(unzipped)
