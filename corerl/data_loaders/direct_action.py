from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from corerl.environment.reward.base import BaseReward
from corerl.data_loaders.base import BaseDataLoader
from corerl.data.data import ObsTransition
from corerl.data.obs_normalizer import ObsTransitionNormalizer


class DirectActionDataLoader(BaseDataLoader):
    """
    This class takes a dataset consisting of a group of CSV files and produces a list of observation transitions
    """

    def __init__(self, cfg: DictConfig):
        self.offline_data_path = Path(cfg.offline_data_path)

        # You can either load all the csvs in the directory or a subset
        if not cfg.train_filenames:
            # will return all files as training data
            self.train_filenames = list(self.offline_data_path.glob('*.csv'))
            self.test_filenames = []
        else:
            self.train_filenames = [self.offline_data_path / file for file in cfg.train_filenames]
            self.test_filenames = [self.offline_data_path / file for file in cfg.test_filenames]

        self.all_filenames = self.train_filenames + self.test_filenames
        # Are we using separate files for the text set
        self.skip_rows = cfg.skip_rows
        self.header = cfg.header
        # not sure if OmegaConf.to_object is necessary anymore?
        self.df_col_names = OmegaConf.to_object(cfg.df_col_names)
        assert len(
            self.df_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'df_col_names', a list of names you'd like to give the columns in your dataframe"
        self.obs_col_names = cfg.obs_col_names
        assert len(
            self.obs_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'obs_col_names', a sublist of the column names in self.df_col_names that you'd like to be included in observations"
        self.action_col_names = OmegaConf.to_object(cfg.action_col_names)
        assert len(
            self.action_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'action_names', a sublist of the column names in self.df_col_names that correspond to the dimensions of the action space"
        self.date_col_name = cfg.date_col_name
        self.max_time_delta = cfg.max_time_delta
        self.time_thresh = pd.Timedelta(self.max_time_delta, "s")
        self.obs_length = cfg.obs_length
        self.steps_per_decision = cfg.steps_per_decision

    def load_data(self, filenames: list[str]) -> pd.DataFrame | None:
        """
        Read csvs into a single concatenated df sorted by date, containing only the columns in the observation space
        """
        if len(filenames) != 0:
            dfs = []
            for file in filenames:
                df = pd.read_csv(file,
                                 dtype=np.float32,
                                 skiprows=self.skip_rows,
                                 header=self.header,
                                 names=self.df_col_names,
                                 index_col=self.date_col_name, parse_dates=True)
                dfs.append(df)

            concat_df = pd.concat(dfs)
            concat_df.sort_values(by=[self.date_col_name], inplace=True)
            concat_df = concat_df[self.obs_col_names + self.action_col_names]
            concat_df = concat_df.ffill()

            return concat_df
        else:
            return None

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
        obs = np.mean(obs, axis=0)
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
        curr_date = None
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

    def create_obs_transitions(self, df: pd.DataFrame,
                               normalizer: ObsTransitionNormalizer,
                               reward_function: BaseReward, *args) -> list[ObsTransition]:
        """
        Iterate through the df and produce transitions using the "Anytime" paradigm.
        Take into account discontinuities in the dataframe (large gaps in time between consecutive rows)
        Creates fixed n-step transitions or variable n-step transitions that always bootstrap off the state at the next decision point

        Will also normalize the observation transitions using the normalizer
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
            obs = np.empty(0)
            prev_decision_point = None
            prev_steps_until_decision = None
            transition_added = False
            prev_obs_transition = None
            while not data_gap and action_start < df_end:
                curr_action, action_end, next_action_start, trunc, term, data_gap = self.find_action_boundary(action_df,
                                                                                                              action_start)

                if data_gap:
                    curr_action_steps, step_start = self.get_curr_action_steps(action_start, action_end)
                else:
                    curr_action_steps, step_start = self.get_curr_action_steps(action_start, next_action_start)

                step_remainder = curr_action_steps % self.steps_per_decision
                steps_until_decision = (step_remainder - 1) % self.steps_per_decision
                if steps_until_decision == 0:
                    steps_until_decision = self.steps_per_decision

                # Adjust the steps_until_decision for the state on the boundary between two different actions
                if prev_obs_transition and curr_action_steps > 0:
                    prev_steps_until_decision = step_remainder
                    if prev_steps_until_decision == 0:
                        prev_steps_until_decision = self.steps_per_decision
                    prev_obs_transition.next_obs_steps_until_decision = prev_steps_until_decision
                    obs_transitions.append(prev_obs_transition)
                    prev_obs_transition = None

                # Ensuring last ObsTransition right before data gap has gap attribute set to True
                if curr_action_steps == 0 and data_gap and len(obs_transitions) > 0:
                    obs_transitions[-1].gap = True
                
                # Next, iterate over current action time steps and produce obs transitions
                for step in range(curr_action_steps):
                    decision_point = steps_until_decision == self.steps_per_decision

                    step_end = step_start + timedelta(seconds=self.obs_length)
                    next_obs = self.get_obs(obs_df, step_start, step_end)

                    # Any way to make the creation of reward_info more universal?
                    reward_info = {}
                    reward_info['prev_action'] = prev_action
                    reward_info['curr_action'] = curr_action
                    reward = reward_function(next_obs, **reward_info)

                    if obs.any():
                        obs_transition = ObsTransition(
                            prev_action,
                            obs,
                            prev_steps_until_decision,
                            prev_decision_point,
                            curr_action,
                            reward,
                            next_obs,
                            steps_until_decision,
                            decision_point,
                            False,  # assume a continuing env
                            False,  # assume a continuing env
                            gap=(step == curr_action_steps - 1) and data_gap  # if the last step and there is a data gap
                        )
                        obs_transition = normalizer.normalize(obs_transition)

                        if step < (curr_action_steps - 1):
                            obs_transitions.append(obs_transition)
                        else:
                            prev_obs_transition = obs_transition

                        if not transition_added:
                            obs_transition.obs_dp = True
                            transition_added = True

                    prev_action = curr_action
                    step_start = step_start + timedelta(seconds=self.obs_length)
                    obs = next_obs
                    prev_decision_point = decision_point
                    prev_steps_until_decision = steps_until_decision
                    steps_until_decision -= 1
                    if steps_until_decision == 0:
                        steps_until_decision = self.steps_per_decision

                    try:
                        pbar.n = df.index.get_loc(step_start)
                        pbar.refresh()
                    except:
                        pass

                action_start = next_action_start

        print("Number of observation transitions: {}".format(len(obs_transitions)))

        return obs_transitions
