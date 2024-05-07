from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm

from datetime import timedelta
import pandas as pd
import numpy as np
import pickle as pkl
import random
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
        # if we return the state constructors's internal state for each transition
        self.return_sc_state = cfg.return_sc_state

    def load_data(self) -> pd.DataFrame:
        """
        Read csvs into a single concatenated df sorted by date, containing only the columns in the observation space
        """
        dfs = []
        for file in self.offline_filenames:
            df = pd.read_csv(file, dtype=np.float32, skiprows=self.skip_rows, header=self.header,
                             names=self.df_col_names, index_col=self.date_col_name, parse_dates=True)
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
                  start_ind: pd.Timestamp) -> (BaseStateConstructor, pd.Timestamp, np.ndarray, bool, BaseStateConstructor):
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
                if step_remainder > 0:
                    steps_since_decision = (self.steps_per_decision - step_remainder) + 1
                else:
                    steps_since_decision = 1
                for i in range(curr_action_steps):
                    decision_point = steps_since_decision == 0
                    _, state, sc_at_state = self.get_state(state_constructor,
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
                  steps_since_decision: int) -> (pd.DataFrame, np.ndarray):

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

        return obs_df, state, sc_at_state

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
                                   transitions: list[tuple],
                                   state_action_rewards: list[tuple],
                                   boot_state: np.ndarray,
                                   term: bool,
                                   trunc: bool) -> list[tuple]:
        # Create "Anytime" variable n-step transitions, where 'n' depends on the number of steps from the action boundary
        n_step_reward = 0.0
        gamma_exp = 1
        for i in range(len(state_action_rewards) - 1, -1, -1):
            sar = state_action_rewards[i]
            state = sar[0]
            action = sar[1]
            reward = sar[2]
            decision_point = sar[3]
            # Recursively updating n-step reward
            n_step_reward = reward + self.gamma * n_step_reward
            transitions.append((state, action, n_step_reward, boot_state, term, trunc, int(decision_point), gamma_exp))
            gamma_exp += 1

        return transitions

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
                           interaction: NormalizerInteraction) -> (list[tuple], list[BaseStateConstructor]):
        """
        Iterate through the df and produce transitions using the "Anytime" paradigm.
        Take into account discontinuities in the dataframe (large gaps in time between consecutive rows)
        """
        transitions = []
        sc_states = []  # will keep a list of the state constructor at each state in transitions

        # Keep trying to create transitions until you reach the end of the df
        action_start = df.iloc[0].name
        df_end = df.iloc[-1].name
        pbar = tqdm(total=df.index.get_loc(df_end))

        while action_start < df_end:
            data_gap = False  # Indicates a discontinuity in the df
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
                curr_action, action_end, next_action_start, trunc, term, data_gap = self.find_action_boundary(df,
                                                                                                              action_start)
                norm_curr_action = interaction.action_normalizer(curr_action)

                # Align time steps within action window
                curr_action_steps, step_start = self.get_curr_action_steps(action_start, action_end)
                step_remainder = curr_action_steps % self.steps_per_decision
                if step_remainder > 0:
                    steps_since_decision = (self.steps_per_decision - step_remainder) + 1
                else:
                    steps_since_decision = 1

                # Iterate over current action time steps and produce (S,A,R)
                state_action_rewards = []

                for i in range(curr_action_steps):
                    decision_point = steps_since_decision == 0
                    obs_df, next_state, sc_at_next_state = self.get_state(state_constructor,
                                                                          interaction,
                                                                          df,
                                                                          step_start,
                                                                          decision_point,
                                                                          steps_since_decision)

                    # Any way to make the creation of reward_info more universal?
                    reward_info = {}
                    reward_info['df'] = obs_df
                    reward_info['prev_action'] = prev_action
                    reward_info['curr_action'] = curr_action
                    raw_reward = reward_function(**reward_info)
                    reward = interaction.action_normalizer(raw_reward)
                    state_action_rewards.append((state, norm_curr_action, reward, prev_decision_point))
                    sc_states.append(sc_at_next_state)

                    # Create n-step Transitions
                    if decision_point and len(state_action_rewards) > 0:
                        boot_state = next_state
                        # Set trunc and term to false since we haven't reached the final action boundary within the action window
                        transitions = self.create_n_step_transitions_(transitions, state_action_rewards, boot_state,
                                                                      False, False)
                        state_action_rewards = []

                    prev_action = curr_action
                    step_start = step_start + timedelta(seconds=self.obs_length)
                    state = next_state
                    steps_since_decision = (steps_since_decision + 1) % self.steps_per_decision
                    prev_decision_point = decision_point

                    try:
                        pbar.n = df.index.get_loc(step_start)
                        pbar.refresh()
                    except:
                        pass

                # Create remaining transitions
                boot_state = state
                transitions = self.create_n_step_transitions_(transitions, state_action_rewards, boot_state, trunc,
                                                              term)

                action_start = next_action_start

        print("Total Transitions:", len(transitions))

        print(len(transitions))
        print(len(sc_states))

        assert len(transitions) == len(sc_states)

        return transitions, sc_states

    def train_test_split(self, transitions: list[tuple], shuffle: bool = True) -> (list[tuple], list[tuple]):
        if shuffle:
            random.shuffle(transitions)

        num_samples = len(transitions)
        train_samples = int(self.train_split * num_samples)
        train_transitions = transitions[:train_samples]
        test_transitions = transitions[train_samples:]

        return train_transitions, test_transitions

    def save_transitions(self, transitions: list[tuple], path: Path):
        with open(path, "wb") as transition_file:
            pkl.dump(transitions, transition_file)

    def load_transitions(self, path: Path) -> list[tuple]:
        with open(path, "rb") as transition_file:
            transitions = pkl.load(transition_file)

            return transitions
