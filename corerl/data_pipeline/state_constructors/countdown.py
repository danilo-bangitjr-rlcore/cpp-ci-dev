from dataclasses import dataclass
from datetime import timedelta
from typing import Type

import numpy as np
import pandas as pd

from corerl.configs.config import config, interpolate
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.utils import get_tag_temporal_state


@config()
class CountdownConfig:
    action_period: timedelta = interpolate('${env.action_period}')
    obs_period: timedelta = interpolate('${env.obs_period}')
    kind: str = 'no_countdown'


@dataclass
class CountdownTS:
    clock: int
    steps_until_dp: int
    last_row: np.ndarray | None


class DecisionPointDetector:
    cd_tag = '__countdown__'

    def __init__(self, cfg: CountdownConfig):
        self._cfg = cfg
        self._steps_per_decision = int(cfg.action_period.total_seconds() / cfg.obs_period.total_seconds())
        assert np.isclose(
            self._steps_per_decision, cfg.action_period.total_seconds() / cfg.obs_period.total_seconds()
        ), "action period must be a multiple of obs period"


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ts = get_tag_temporal_state(
            stage=StageCode.SC,
            tag=self.cd_tag,
            ts=pf.temporal_state,
            default=lambda: CountdownTS(
                clock=self._steps_per_decision - 1,
                steps_until_dp=0,
                last_row=None,
            ),
        )

        if ts.last_row is None:
            ts = self._warmup_ts(pf.actions, ts)

        n_rows = len(pf.data)
        clock_feats = self._init_feature_builder(n_rows)

        for i in range(n_rows):
            is_dp = ts.steps_until_dp == 0
            is_ac = self._is_action_change(pf.actions, ts, i)

            if is_dp or is_ac:
                pf.decision_points[i] = True
                ts.steps_until_dp = self._steps_per_decision

            clock_feats.tick(i, ts.clock, ts.steps_until_dp)

            # loop carry state
            ts.clock = (ts.clock - 1) % self._steps_per_decision
            ts.steps_until_dp -= 1

        # special case if no countdown features are needed
        if isinstance(clock_feats, NoCountdown):
            return pf

        # otherwise add features to df
        clock_representation = clock_feats.get()
        n_clock_feats = clock_representation.shape[1]
        for feat_col in range(n_clock_feats):
            pf.data[f'countdown.[{feat_col}]'] = clock_representation[:, feat_col]

        return pf


    def _warmup_ts(self, df: pd.DataFrame, ts: CountdownTS):
        """
        Look forward in time for first action change. Use that
        to set the starting point for the clock.
        """
        n_rows = len(df)

        for i in range(n_rows):
            is_ac = self._is_action_change(df, ts, i)
            if is_ac:
                ts.steps_until_dp = i % self._steps_per_decision
                ts.clock = (i - 1) % self._steps_per_decision
                break

        # make sure we haven't mutated irrelevant
        # ts states
        ts.last_row = None
        return ts

    def _is_action_change(self, actions: pd.DataFrame, ts: CountdownTS, idx: int):
        # define the no action case as never having an action change
        if len(actions.columns) == 0:
            return False

        if ts.last_row is None:
            ts.last_row = actions.iloc[0].to_numpy()

        row = actions.iloc[idx].to_numpy()
        is_ac = not np.all(ts.last_row == row)
        ts.last_row = row

        return is_ac


    def _init_feature_builder(self, n_rows: int):
        builders: dict[str, Type[CountdownFeatureBuilder]] = {
            'no_countdown': NoCountdown,
            'two_clock': TwoClockCountdown,
            'one_hot': OneHotCountdown,
            'int': IntCountdown,
        }

        builder = builders.get(self._cfg.kind)
        assert builder is not None, f'Unknown type of action period countdown features: {self._cfg.kind}'

        return builder(n_rows, self._steps_per_decision)


# --------------------------------
# -- Countdown Feature Builders --
# --------------------------------
class CountdownFeatureBuilder:
    def __init__(self, n_rows: int, period: int):
        self._period = period

    def tick(self, row: int, clock: int, steps_until_dp: int) -> None:
        raise NotImplementedError()

    def get(self) -> np.ndarray:
        raise NotImplementedError()


class NoCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        super().__init__(n_rows, period)

    def tick(self, row: int, clock: int, steps_until_dp: int) -> None:
        ...


class TwoClockCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        super().__init__(n_rows, period)
        self._x = np.zeros((n_rows, 2), dtype=np.int_)

    def tick(self, row: int, clock: int, steps_until_dp: int):
        # shift clock to be [1, period]
        # instead of        [0, period)
        self._x[row, 0] = clock + 1
        self._x[row, 1] = steps_until_dp

    def get(self):
        return self._x


class OneHotCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        super().__init__(n_rows, period)
        self._x = np.zeros((n_rows, period), dtype=np.bool_)

    def tick(self, row: int, clock: int, steps_until_dp: int):
        hot = steps_until_dp - 1
        self._x[row, hot] = 1

    def get(self):
        return self._x


class IntCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        super().__init__(n_rows, period)
        self._x = np.zeros((n_rows, 1), dtype=np.int_)

    def tick(self, row: int, clock: int, steps_until_dp: int):
        self._x[row, 0] = steps_until_dp

    def get(self):
        return self._x
