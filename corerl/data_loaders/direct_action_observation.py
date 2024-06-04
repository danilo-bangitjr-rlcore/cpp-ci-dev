from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from datetime import timedelta
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm

from corerl.environment.reward.base import BaseReward
from corerl.data_loaders.base import BaseDataLoader
from corerl.data_loaders.utils import ObsTransition


class DirectActionDataLoader(BaseDataLoader):
    def __init__(self, cfg: DictConfig):
        self.offline_data_path = Path(cfg.offline_data_path)
        # You can either load all the csvs in the directory or a subset
        if cfg.train_filenames == []:
            self.train_filenames = list(self.offline_data_path.glob('*.csv'))
        else:
            self.train_filenames = [self.offline_data_path / file for file in cfg.train_filenames]

        # Are we using separate files for the text set
        self.skip_rows = cfg.skip_rows
        self.header = cfg.header
        # not sure if OmegaConf.to_object is necessary anymore?
        self.df_col_names = OmegaConf.to_object(cfg.df_col_names)

        self.endo_obs_col_names = OmegaConf.to_object(cfg.endo_obs_col_names)
        self.num_endo = len(self.endo_obs_col_names)  # the numer of endoenous variables
        self.exo_obs_col_names = OmegaConf.to_object(cfg.exo_obs_col_names)
        self.separate_endo_exo = cfg.separate_endo_exo
        self.obs_col_names = self.endo_obs_col_names + self.exo_obs_col_names
        self.action_col_names = OmegaConf.to_object(cfg.action_col_names)

        self.date_col_name = cfg.date_col_name
        self.max_time_delta = cfg.max_time_delta
        self.time_thresh = pd.Timedelta(self.max_time_delta, "s")
        self.steps_per_decision = cfg.steps_per_decision
        self.obs_length = cfg.obs_length
        self.warmup_steps = cfg.warmup_steps
        self.gamma = cfg.gamma
        self.train_split = cfg.train_split
        self.n_step = cfg.n_step
        self.trajectory_len = cfg.trajectory_len
        self.return_test_trajectory = self.trajectory_len != 0

        # if we return the state constructor's internal state for each transition
        self.return_sc_state = cfg.return_sc_state

    def load_data(self) -> pd.DataFrame:
        """
        Read csvs into a single concatenated df sorted by date, containing only the columns in the observation space
        """
        dfs = []
        for file in self.train_filenames:
            df = pd.read_csv(file, dtype=np.float32, skiprows=self.skip_rows, header=self.header,
                             names=self.df_col_names, index_col=self.date_col_name, parse_dates=True)
            dfs.append(df)
        concat_df = pd.concat(dfs)
        concat_df.sort_values(by=[self.date_col_name], inplace=True)
        concat_df = concat_df[self.obs_col_names + self.action_col_names]
        concat_df = concat_df.ffill()

        print("Concatenated DF:")
        print(concat_df)

        return concat_df

    def get_obs_max_min(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Find the max and min values for each column in the input df to later be used for normalization
        """

        obs_df = df[self.obs_col_names]
        np_min_max = obs_df.agg(['min', 'max']).to_numpy()
        obs_space_low = np_min_max[0, :]
        obs_space_high = np_min_max[1, :]

        print("Obs Space Low:", obs_space_low)
        print("Obs Space High:", obs_space_high)

        return obs_space_low, obs_space_high

    def get_df_date_range(self, df: pd.DataFrame, start_ind: pd.Timestamp, end_ind: pd.Timestamp) -> pd.DataFrame:
        window_df = df.loc[start_ind: end_ind]
        return window_df

    def get_obs(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        obs_df = self.get_df_date_range(df, start, end)  # here it is only used for observations
        obs = obs_df.to_numpy()


        try:
            obs = np.mean(obs, axis=0)
        except RuntimeWarning:
            print(obs)
            print("invalie")
            assert False



        return obs

    def find_action_boundary(self, action_df: pd.DataFrame,
                             start_ind: pd.Timestamp) -> (np.ndarray, pd.Timestamp, pd.Timestamp, bool, bool, bool):
        """
        Return the action taken at the beginning of the dataframe.
        Iterate through the dataframe until an action change, a truncation/termination in the episode, or a large break in time.
        Return the timestamp of the action change/truncation/termination/break in time and whether there was a truncation or termination
        """
        data_gap = False
        prev_date = start_ind
        curr_action = action_df.loc[start_ind].to_numpy()
        for curr_date, row in action_df.loc[start_ind:].iterrows():
            date_diff = curr_date - prev_date
            # Is there a large gap in time between consecutive rows in the df?
            if date_diff > self.time_thresh:
                data_gap = True
            # Is the episode terminated/truncated?
            trunc, term = self.check_termination_truncation(row)
            row_action = row[self.action_col_names].to_numpy()
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

    def get_curr_action_steps(self, action_start: pd.Timestamp, action_end: pd.Timestamp) -> (int, pd.Timestamp):
        """
        Determine the number of time steps that fit in the given action window and align the time steps, taking into account the time remainder
        """
        action_duration = action_end - action_start
        curr_action_steps = action_duration // timedelta(seconds=self.obs_length)
        time_remainder = action_duration % timedelta(seconds=self.obs_length)
        step_start = action_start + time_remainder

        return curr_action_steps, step_start

    def create_obs_transitions(self, df: pd.DataFrame, reward_function: BaseReward) -> list[ObsTransition]:

        """
        Iterate through the df and produce transitions using the "Anytime" paradigm.
        Take into account discontinuities in the dataframe (large gaps in time between consecutive rows)
        Creates fixed n-step transitions or variable n-step transitions that always bootstrap off the state at the next decision point
        """

        obs_transitions = []

        # Keep trying to create transitions until you reach the end of the df
        action_start = df.iloc[0].name
        df_end = df.iloc[-1].name
        pbar = tqdm(total=df.index.get_loc(df_end))

        action_df = df[self.action_col_names]
        obs_df = df[self.obs_col_names]

        while action_start < df_end:
            data_gap = False  # Indicates a discontinuity in the df
            prev_action = None
            while not data_gap and action_start < df_end:
                curr_action, action_end, next_action_start, trunc, term, data_gap = self.find_action_boundary(action_df,
                                                                                                              action_start)
                # Align time steps within action window
                curr_action_steps, step_start = self.get_curr_action_steps(action_start, action_end)
                # Next, iterate over current action time steps and produce partial transitions
                if curr_action_steps > 0:
                    # produce the initial observation
                    step_end = step_start + timedelta(seconds=self.obs_length)
                    last_obs = self.get_obs(obs_df, step_end, step_end)
                    step_start = step_start + timedelta(seconds=self.obs_length)

                    # how many steps we are talking the current action, minus the first observation
                    for step in range(curr_action_steps - 1):
                        step_end = step_start + timedelta(seconds=self.obs_length)
                        obs = self.get_obs(obs_df, step_end, step_end)

                        # Any way to make the creation of reward_info more universal?
                        reward_info = {}
                        reward_info['prev_action'] = prev_action
                        reward_info['curr_action'] = curr_action
                        reward = reward_function(obs, **reward_info)

                        transition = ObsTransition(
                            last_obs,
                            curr_action,
                            reward,
                            obs,
                            False,  # assume a continuing env
                            False,  # assume a continuing env
                            gap=(step == curr_action_steps - 2) and data_gap  # if the last step and there is a data gap
                        )

                        if (step == curr_action_steps - 2) and data_gap:
                            print("GAP!")

                        obs_transitions.append(transition)

                        prev_action = curr_action
                        step_start = step_start + timedelta(seconds=self.obs_length)
                        last_obs = obs

                        try:
                            pbar.n = df.index.get_loc(step_start)
                            pbar.refresh()
                        except:
                            pass

                action_start = next_action_start

        print("Number of observation transitions: {}".format(len(obs_transitions)))

        return obs_transitions

    def create_transitions(self,
                           df: pd.DataFrame,
                           reward_function: BaseReward, *args) -> dict:
        pass
