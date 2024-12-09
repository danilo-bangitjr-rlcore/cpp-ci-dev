from typing import Type
import numpy as np

from collections.abc import Hashable
from dataclasses import dataclass

from corerl.data_pipeline.utils import get_tag_temporal_state
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.hydra import interpolate


@dataclass
class CountdownConfig:
    action_period: int = interpolate('${action_period}')
    kind: str = 'one_hot'



@dataclass
class CountdownTS:
    clock: int
    steps_since_dp: int


class CountdownAdder:
    cd_tag = '__countdown__'

    def __init__(self, tag_cfgs: list[TagConfig], cfg: CountdownConfig):
        self._cfg = cfg

        # have to widen the type here because pandas...
        # df.get uses the invariant `list[T]` type for
        # its argument annotations
        self._action_tags: list[Hashable] = [
            tag.name
            for tag in tag_cfgs
            if tag.is_action
        ]


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ts = get_tag_temporal_state(
            stage=StageCode.SC,
            tag=self.cd_tag,
            ts=pf.temporal_state,
            default=lambda: CountdownTS(
                clock=self._cfg.action_period - 1,
                steps_since_dp=0,
            ),
        )

        action_rows = pf.data.get(self._action_tags)
        assert action_rows is not None

        n_rows = len(action_rows)
        clock_feats = self._init_feature_builder(n_rows)

        last_row = action_rows.iloc[0].to_numpy()
        for i in range(n_rows):
            row = action_rows.iloc[i].to_numpy()

            is_action_change = not np.all(last_row == row)
            is_dp = ts.steps_since_dp == 0

            if is_action_change or is_dp:
                ts.steps_since_dp = self._cfg.action_period

            clock_feats.tick(i, ts.clock, ts.steps_since_dp)

            # loop carry state
            ts.clock = (ts.clock - 1) % self._cfg.action_period
            ts.steps_since_dp -= 1
            last_row = row

        clock_representation = clock_feats.get()
        n_clock_feats = clock_representation.shape[1]
        for feat_col in range(n_clock_feats):
            pf.data[f'countdown.[{feat_col}]'] = clock_representation[:, feat_col]

        return pf


    def _init_feature_builder(self, n_rows: int):
        builders: dict[str, Type[CountdownFeatureBuilder]] = {
            'two_clock': TwoClockCountdown,
            'one_hot': OneHotCountdown,
            'int': IntCountdown,
        }

        builder = builders.get(self._cfg.kind)
        assert builder is not None, f'Unknown type of action period countdown features: {self._cfg.kind}'

        return builder(n_rows, self._cfg.action_period)


# --------------------------------
# -- Countdown Feature Builders --
# --------------------------------
class CountdownFeatureBuilder:
    def __init__(self, n_rows: int, period: int):
        self._period = period

    def tick(self, row: int, clock: int, steps_since_dp: int) -> None:
        ...

    def get(self) -> np.ndarray:
        ...


class TwoClockCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        self._period = period
        self._x = np.zeros((n_rows, 2), dtype=np.int_)

    def tick(self, row: int, clock: int, steps_since_dp: int):
        # shift clock to be [1, period]
        # instead of        [0, period)
        self._x[row, 0] = clock + 1
        self._x[row, 1] = steps_since_dp

    def get(self):
        return self._x


class OneHotCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        self._period = period
        self._x = np.zeros((n_rows, period), dtype=np.bool_)

    def tick(self, row: int, clock: int, steps_since_dp: int):
        hot = steps_since_dp - 1
        self._x[row, hot] = 1

    def get(self):
        return self._x


class IntCountdown(CountdownFeatureBuilder):
    def __init__(self, n_rows: int, period: int):
        self._period = period
        self._x = np.zeros((n_rows, 1), dtype=np.int_)

    def tick(self, row: int, clock: int, steps_since_dp: int):
        self._x[row, 0] = steps_since_dp

    def get(self):
        return self._x
