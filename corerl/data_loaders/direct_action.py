import logging
from collections.abc import Sequence
from omegaconf import DictConfig
from pathlib import Path

from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from corerl.environment.reward.base import BaseReward
from corerl.data_loaders.base import BaseDataLoader, OldBaseDataLoader
from corerl.data.data import OldObsTransition, ObsTransition
from corerl.data.obs_normalizer import ObsTransitionNormalizer

from warnings import warn

log = logging.getLogger(__name__)


class OldDirectActionDataLoader(OldBaseDataLoader):
    """
    This class takes a dataset consisting of a group of CSV files and produces a list of observation transitions
    """

    def __init__(self, cfg: DictConfig):

        warn("You are using a deprecated version of the DirectActionDataLoader", stacklevel=1)

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
        self.df_col_names = cfg.df_col_names
        assert len(self.df_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'df_col_names', a list of names you'd like to give the columns in your dataframe" # noqa: E501
        self.obs_col_names = cfg.obs_col_names
        assert len(self.obs_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'obs_col_names', a sublist of the column names in self.df_col_names that you'd like to be included in observations" # noqa: E501
        self.action_col_names = cfg.action_col_names
        assert len(self.action_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'action_names', a sublist of the column names in self.df_col_names that correspond to the dimensions of the action space" # noqa: E501
        self.date_col_name = cfg.date_col_name
        self.max_time_delta = cfg.max_time_delta
        self.time_thresh = pd.Timedelta(self.max_time_delta, "s")
        self.obs_length = cfg.obs_length
        self.steps_per_decision = cfg.steps_per_decision

    def load_data(
        self,
        filenames: Sequence[str] | Sequence[Path],
    ) -> pd.DataFrame:
        """
        Read csvs into a single concatenated df sorted by date, containing only the columns in the observation space
        """
        if len(filenames) == 0:
            return pd.DataFrame()

        dfs = []
        for file in filenames:
            df = pd.read_csv(
                file,
                dtype=np.float32,
                skiprows=self.skip_rows,
                header=self.header,
                names=self.df_col_names,
                index_col=self.date_col_name,
                parse_dates=True,
            )
            dfs.append(df)

        concat_df = pd.concat(dfs)
        concat_df.sort_values(by=[self.date_col_name], inplace=True)
        concat_df = concat_df[self.obs_col_names + self.action_col_names]
        concat_df = concat_df.ffill()

        return concat_df

    def get_obs_max_min(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the max and min values for each column in the input df to later be used for normalization
        """

        obs_df = df[self.obs_col_names]
        np_min_max = obs_df.agg(['min', 'max']).to_numpy()
        obs_space_low = np_min_max[0, :]
        obs_space_high = np_min_max[1, :]

        log.debug(f"Obs Space Low: {obs_space_low}")
        log.debug(f"Obs Space High: {obs_space_high}")

        return obs_space_low, obs_space_high

    def get_df_date_range(self, df: pd.DataFrame, start_ind: pd.Timestamp, end_ind: pd.Timestamp) -> pd.DataFrame:
        window_df = df.loc[start_ind: end_ind]
        return window_df

    def get_obs(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        obs_df = self.get_df_date_range(df, start, end)  # here it is only used for observations
        obs = obs_df.to_numpy()
        obs = np.mean(obs, axis=0)
        return obs

    def find_action_boundary(
        self,
        action_df: pd.DataFrame,
        start_ind: pd.Timestamp,
    ) -> tuple[np.ndarray, pd.Timestamp, pd.Timestamp, bool]:
        """
        Return the action taken at the beginning of the dataframe.
        Iterate through the dataframe until an action change,
        a truncation/termination in the episode, or a large break in time.
        Return the timestamp of the action change/truncation/termination/break
        in time and whether there was a truncation or termination
        """
        data_gap = False
        prev_date = start_ind
        curr_action = action_df.loc[start_ind].to_numpy()
        curr_date = None

        new_action = None
        new_action_start = None

        for curr_date, row in action_df.loc[start_ind:].iterrows():
            assert isinstance(curr_date, pd.Timestamp)
            date_diff = curr_date - prev_date
            # Is there a large gap in time between consecutive rows in the df?
            if date_diff > self.time_thresh:
                data_gap = True
            # Is the episode terminated/truncated?
            row_action = row[self.action_col_names].to_numpy()

            # have to do some business with actions and such
            if not np.array_equal(curr_action, row_action):  # if the row action is not equal to the current action
                if row_action != new_action:
                    new_action_start = curr_date
                    new_action = row_action

                # if the new action has been held for more than obs_length
                assert new_action_start is not None
                if curr_date - new_action_start >= timedelta(seconds=self.obs_length):
                    new_action = True

            # Has there been a change in action/truncation/termination/gap in data?
            if data_gap or new_action:
                return curr_action, prev_date, curr_date, data_gap

            prev_date = curr_date

        assert curr_date is not None
        return curr_action, prev_date, curr_date, False

    def get_curr_action_steps(
        self,
        action_start: pd.Timestamp,
        action_end: pd.Timestamp,
    ) -> tuple[int, pd.Timestamp]:
        """
        Determine the number of time steps that fit in the given action window and align the time steps,
        taking into account the time remainder
        """
        action_duration = action_end - action_start
        curr_action_steps = action_duration // timedelta(seconds=self.obs_length)
        time_remainder = action_duration % timedelta(seconds=self.obs_length)
        step_start = action_start + time_remainder

        return curr_action_steps, step_start

    def create_obs_transitions(
        self,
        df: pd.DataFrame,
        normalizer: ObsTransitionNormalizer,
        reward_function: BaseReward,
        *args,
    ) -> list[OldObsTransition]:
        """
        Iterate through the df and produce transitions using the "Anytime" paradigm.
        Take into account discontinuities in the dataframe (large gaps in time between consecutive rows)
        Creates fixed n-step transitions or variable n-step transitions
        that always bootstrap off the state at the next decision point

        Will also normalize the observation transitions using the normalizer
        """
        obs_transitions = []

        # Keep trying to create transitions until you reach the end of the df
        action_start = df.iloc[0].name
        df_end = df.iloc[-1].name
        pbar = tqdm(total=len(df))

        action_df = df[self.action_col_names]
        obs_df = df[self.obs_col_names]

        prev_obs_transition: OldObsTransition | None = None

        debug_idx = 0

        assert isinstance(action_start, pd.Timestamp)
        assert isinstance(df_end, pd.Timestamp)

        while action_start < df_end:
            data_gap = False  # Indicates a discontinuity in the df
            prev_action = None
            obs = np.empty(0)
            prev_decision_point = None
            prev_steps_until_decision = None
            transition_added = False

            while not data_gap and action_start < df_end:
                (   curr_action,
                    action_end,
                    next_action_start,
                    data_gap,
                ) = self.find_action_boundary(action_df, action_start)
                action_transitions = []

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
                    if prev_obs_transition:
                        obs_transitions.append(prev_obs_transition)
                        prev_obs_transition = None
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
                        assert prev_steps_until_decision is not None
                        assert prev_decision_point is not None
                        obs_transition = OldObsTransition(
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
                        assert isinstance(obs_transition, OldObsTransition)

                        # Make first obs in chunk a decision point
                        if not transition_added:
                            obs_transition.obs_dp = True
                            transition_added = True

                        if len(obs_transitions) > 0 and not obs_transitions[-1].gap:
                            l_next_obs = obs_transition.obs
                            assert l_next_obs is not None
                            assert np.allclose(l_next_obs, obs_transitions[-1].next_obs)

                        if step < (curr_action_steps - 1):
                            obs_transitions.append(obs_transition)
                            action_transitions.append(obs_transition)
                        else:
                            prev_obs_transition = obs_transition

                    debug_idx += 1

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
                    except Exception:
                        pass

                if len(action_transitions) > 0:
                    assert action_transitions[0].obs_dp
                action_start = next_action_start

        chunks = []
        curr_chunk = []
        for obs_transition in obs_transitions:
            curr_chunk.append(obs_transition)
            if obs_transition.gap:
                chunks.append(curr_chunk)
                curr_chunk = []

        chunks.append(curr_chunk)

        assert len(obs_transitions) == sum([len(chunk) for chunk in chunks])
        fixed_transitions = []
        for chunk in chunks:
            fixed_transitions += self.fix_obs_transitions(chunk)

        assert len(fixed_transitions) == len(obs_transitions)

        return fixed_transitions

    def fix_obs_transitions(self, obs_transitions):
        action_windows = get_action_windows(obs_transitions)
        # pre-compute the steps until decisions
        # this is a list of the steps per decision for the initial obs in each obs transition
        steps_until_decisions = []
        dps = []
        for curr_action_obs_transitions in action_windows:
            num_action_steps = len(curr_action_obs_transitions)  # number of steps where that action was taken
            remainder_steps = (num_action_steps % self.steps_per_decision)
            steps_until_decision = remainder_steps
            initial = True

            for _ in curr_action_obs_transitions:
                dp = False
                if steps_until_decision == 0:
                    steps_until_decision = self.steps_per_decision
                    dp = True
                elif initial:
                    dp = True

                dps.append(dp)
                steps_until_decisions.append(steps_until_decision)
                steps_until_decision -= 1
                initial = False

        last_steps_until_decision = steps_until_decisions[-1] - 1
        if last_steps_until_decision == 0:
            last_steps_until_decision = self.steps_per_decision
        elif last_steps_until_decision == -1:
            last_steps_until_decision = self.steps_per_decision - 1

        steps_until_decisions.append(last_steps_until_decision)
        dps.append(last_steps_until_decision == self.steps_per_decision)

        for i in range(len(obs_transitions)):
            obs_transitions[i].obs_steps_until_decision = steps_until_decisions[i]
            obs_transitions[i].next_obs_steps_until_decision = steps_until_decisions[i + 1]

            obs_transitions[i].obs_dp = dps[i]
            obs_transitions[i].next_obs_dp = dps[i + 1]

        return obs_transitions


def get_action_windows(obs_transitions: list[OldObsTransition]):
    curr_action = obs_transitions[0].action  # the first action
    action_windows = []
    curr_action_obs_transitions = []

    for obs_transition in obs_transitions:
        if obs_transition.action == curr_action:
            curr_action_obs_transitions.append(obs_transition)

        else:  # the action changed
            action_windows.append(curr_action_obs_transitions)
            curr_action_obs_transitions = [obs_transition]
            curr_action = obs_transition.action

    action_windows.append(curr_action_obs_transitions)  # append any left-overs as a new action window

    # check that everything is correct
    assert sum([len(aw) for aw in action_windows]) == len(obs_transitions)  # is a partition
    prev_action = None
    for action_window in action_windows:
        action = action_window[0].action
        for obs_t in action_window:  # all elements of the action window has the same action
            assert action == obs_t.action
        assert action != prev_action  # neighbouring action windows have different actions
        prev_action = action

    return action_windows


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
        self.df_col_names = cfg.df_col_names
        assert len(self.df_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'df_col_names', a list of names you'd like to give the columns in your dataframe" # noqa: E501
        self.obs_col_names = cfg.obs_col_names
        assert len(self.obs_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'obs_col_names', a sublist of the column names in self.df_col_names that you'd like to be included in observations" # noqa: E501
        self.action_col_names = cfg.action_col_names
        assert len(self.action_col_names) > 0, "Ensure config/env/<env_name>.yaml defines 'action_names', a sublist of the column names in self.df_col_names that correspond to the dimensions of the action space" # noqa: E501
        self.date_col_name = cfg.date_col_name
        self.max_time_delta = cfg.max_time_delta
        self.time_thresh = pd.Timedelta(self.max_time_delta, "s")
        self.obs_length = cfg.obs_length

    def load_data(
        self,
        filenames: Sequence[str] | Sequence[Path],
    ) -> pd.DataFrame:
        """
        Read csvs into a single concatenated df sorted by date, containing only the columns in the observation space
        """
        if len(filenames) == 0:
            return pd.DataFrame()

        dfs = []
        for file in filenames:
            df = pd.read_csv(
                file,
                dtype=np.float32,
                skiprows=self.skip_rows,
                header=self.header,
                names=self.df_col_names,
                index_col=self.date_col_name,
                parse_dates=True,
            )
            dfs.append(df)

        concat_df = pd.concat(dfs)
        concat_df.sort_values(by=[self.date_col_name], inplace=True)
        concat_df = concat_df[self.obs_col_names + self.action_col_names]
        concat_df = concat_df.ffill()

        return concat_df

    def get_obs_max_min(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the max and min values for each column in the input df to later be used for normalization
        """

        obs_df = df[self.obs_col_names]
        np_min_max = obs_df.agg(['min', 'max']).to_numpy()
        obs_space_low = np_min_max[0, :]
        obs_space_high = np_min_max[1, :]

        log.debug(f"Obs Space Low: {obs_space_low}")
        log.debug(f"Obs Space High: {obs_space_high}")

        return obs_space_low, obs_space_high

    def get_df_date_range(self, df: pd.DataFrame, start_ind: pd.Timestamp, end_ind: pd.Timestamp) -> pd.DataFrame:
        window_df = df.loc[start_ind: end_ind]
        return window_df

    def get_obs(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        obs_df = self.get_df_date_range(df, start, end)  # here it is only used for observations
        obs = obs_df.to_numpy()
        obs = np.mean(obs, axis=0)
        return obs

    def find_action_boundary(
        self,
        action_df: pd.DataFrame,
        start_ind: pd.Timestamp,
    ) -> tuple[np.ndarray, pd.Timestamp, pd.Timestamp, bool, bool, bool]:
        """
        Return the action taken at the beginning of the dataframe.
        Iterate through the dataframe until an action change,
        a truncation/termination in the episode, or a large break in time.
        Return the timestamp of the action change/truncation/termination/break
        in time and whether there was a truncation or termination
        """
        data_gap = False
        prev_date = start_ind
        curr_action = action_df.loc[start_ind].to_numpy()
        curr_date = None
        for curr_date, row in action_df.loc[start_ind:].iterrows():
            assert isinstance(curr_date, pd.Timestamp)
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

        assert curr_date is not None
        return curr_action, prev_date, curr_date, False, False, False

    def check_termination_truncation(self, df: pd.Series) -> tuple[bool, bool]:
        """
        Project specific. Assuming a continuing environment.
        """
        return False, False

    def get_curr_action_steps(self, action_start: pd.Timestamp, action_end: pd.Timestamp) -> tuple[int, pd.Timestamp]:
        """
        Determine the number of time steps that fit in the given action window and align the time steps,
        taking into account the time remainder
        """
        action_duration = action_end - action_start
        curr_action_steps = action_duration // timedelta(seconds=self.obs_length)
        time_remainder = action_duration % timedelta(seconds=self.obs_length)
        step_start = action_start + time_remainder

        return curr_action_steps, step_start

    def create_obs_transitions(
        self,
        df: pd.DataFrame,
        reward_function: BaseReward,
        *args,
    ) -> list[ObsTransition]:
        """
        Iterate through the df and produce observation transitions for that dataframe.
        Takes into account discontinuities in the dataframe (large gaps in time between consecutive rows)
        """
        obs_transitions = []

        # Keep trying to create transitions until you reach the end of the df
        action_start = df.iloc[0].name
        df_end = df.iloc[-1].name
        pbar = tqdm(total=len(df))

        action_df = df[self.action_col_names]
        obs_df = df[self.obs_col_names]

        assert isinstance(action_start, pd.Timestamp)
        assert isinstance(df_end, pd.Timestamp)

        prev_obs_transition = None
        while action_start < df_end:
            data_gap = False  # Indicates a discontinuity in the df
            prev_action = None
            obs = np.empty(0)

            while not data_gap and action_start < df_end:
                (   curr_action,
                    action_end,
                    next_action_start,
                    _, _,
                    data_gap,
                ) = self.find_action_boundary(action_df, action_start)

                if data_gap:
                    curr_action_steps, step_start = self.get_curr_action_steps(action_start, action_end)
                else:
                    curr_action_steps, step_start = self.get_curr_action_steps(action_start, next_action_start)

                if prev_obs_transition and curr_action_steps > 0:
                    obs_transitions.append(prev_obs_transition)
                    prev_obs_transition = None

                # Ensuring last ObsTransition right before data gap has gap attribute set to True
                if curr_action_steps == 0 and data_gap and len(obs_transitions) > 0:
                    if prev_obs_transition:
                        obs_transitions.append(prev_obs_transition)
                        prev_obs_transition = None
                    obs_transitions[-1].gap = True

                # Next, iterate over current action time steps and produce obs transitions
                for step in range(curr_action_steps):
                    step_end = step_start + timedelta(seconds=self.obs_length)
                    next_obs = self.get_obs(obs_df, step_start, step_end)

                    # Any way to make the creation of reward_info more universal?
                    reward_info = {}
                    reward_info['prev_action'] = prev_action
                    reward_info['curr_action'] = curr_action
                    reward = reward_function(next_obs, **reward_info)

                    if obs.any():
                        obs_transition = ObsTransition(
                            obs,
                            curr_action,
                            reward,
                            next_obs,
                            False,  # assume a continuing env
                            False,  # assume a continuing env
                            gap=(step == curr_action_steps - 1) and data_gap  # if the last step and there is a data gap
                        )

                        if len(obs_transitions) > 0 and not obs_transitions[-1].gap:
                            assert np.allclose(obs_transition.obs, obs_transitions[-1].next_obs)

                        if step < (curr_action_steps - 1):
                            obs_transitions.append(obs_transition)
                        else:
                            prev_obs_transition = obs_transition

                    prev_action = curr_action
                    step_start = step_start + timedelta(seconds=self.obs_length)
                    obs = next_obs

                    try:
                        pbar.n = df.index.get_loc(step_start)
                        pbar.refresh()
                    except Exception:
                        pass

                action_start = next_action_start

        log.debug(f"Number of observation transitions: {len(obs_transitions)}")
        return obs_transitions
