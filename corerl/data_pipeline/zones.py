import logging

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.maybe import Maybe

logger = logging.getLogger(__name__)


class ZoneDiscourager:
    def __init__(self, app_state: AppState, tag_configs: list[TagConfig], prep_stage: Preprocessor):
        self._app_state = app_state
        self._prep_stage = prep_stage

        self._bounded_tag_cfgs = [
            tag_cfg for tag_cfg in tag_configs
            if tag_cfg.yellow_bounds is not None
            or tag_cfg.red_bounds is not None
        ]


    def __call__(self, pf: PipelineFrame):
        df = self._prep_stage.inverse(pf.data)

        penalties = np.zeros(df.shape[0])
        for i, (_, row_series) in enumerate(df.iterrows()):
            row = row_series.to_frame().transpose()

            penalties[i] = self._get_penalty_for_row(row)


        # 2025-03-01: put zone violation reward penalties behind feature flag
        if self._app_state.cfg.feature_flags.zone_violations:
            pf.rewards['reward'] += penalties

        return pf


    def _get_penalty_for_row(self, row: pd.DataFrame):
        red_violations = [
            self._red_violation_percent(row, tag)
            for tag in self._bounded_tag_cfgs
        ]

        # first check for red zone violations
        idx = _argmax(red_violations)
        if idx is not None:
            tag = self._bounded_tag_cfgs[idx]
            percent = red_violations[idx]
            assert percent is not None
            return self._handle_red_violation(tag, percent)

        # then yellow zone
        yellow_violations = [
            self._yellow_violation_percent(row, tag)
            for tag in self._bounded_tag_cfgs
        ]

        idx = _argmax(yellow_violations)
        if idx is not None:
            tag = self._bounded_tag_cfgs[idx]
            percent = yellow_violations[idx]
            assert percent is not None
            return self._handle_yellow_violation(tag, percent)

        return 0


    def _handle_yellow_violation(self, tag: TagConfig, percent: float):
        """
        Lifecycle method to handle yellow zone violations. Computes the reward penalty
        given degree of violation.

        Reward penalty ramps slowly from 0 to -2.
        """
        self._app_state.event_bus.emit_event(EventType.yellow_zone_violation)
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric='yellow_zone_violation',
            value=percent,
        )
        logger.warning(f"Yellow zone violation for tag {tag.name} at level: {percent}")
        return -2 * (percent**2)


    def _handle_red_violation(self, tag: TagConfig, percent: float):
        """
        Lifecycle method to handle red zone violations. Computes the reward penalty
        given degree of violation.

        Reward penalty ramps quickly from -4 to -8.
        """
        self._app_state.event_bus.emit_event(EventType.red_zone_violation)
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric='red_zone_violation',
            value=percent,
        )
        logger.error(f"Red zone violation for tag {tag.name} at level: {percent}")
        return -4 - (4 * percent)


    def _yellow_violation_percent(self, row: pd.DataFrame, tag: TagConfig):
        x: float = row[tag.name].to_numpy()[0]

        yellow_lo = (
            Maybe(tag.yellow_bounds)
            .map(lambda bounds: bounds[0])
            .unwrap()
        )
        yellow_hi = (
            Maybe(tag.yellow_bounds)
            .map(lambda bounds: bounds[1])
            .unwrap()
        )

        if yellow_lo is not None and x < yellow_lo:
            next_lo = (
                # the next lowest bound is either the red zone if one exists
                Maybe(tag.red_bounds and tag.red_bounds[0])
                # or the operating bound if one exists
                .otherwise(lambda: tag.operating_range and tag.operating_range[0])
                # and if neither exists, we're in trouble
                .expect(f'Yellow zone specified for tag {tag.name}, but no lower bound found')
            )

            return (yellow_lo - x) / (yellow_lo - next_lo)

        if yellow_hi is not None and x > yellow_hi:
            next_hi = (
                # the next highest bound is either the red zone if one exists
                Maybe(tag.red_bounds and tag.red_bounds[1])
                # or the operating bound if one exists
                .otherwise(lambda: tag.operating_range and tag.operating_range[1])
                # and if neither exists, we're in trouble
                .expect(f'Yellow zone specified for tag {tag.name}, but no upper bound found')
            )

            return (x - yellow_hi) / (next_hi - yellow_hi)

        return None

    def _red_violation_percent(self, row: pd.DataFrame, tag: TagConfig):
        x: float = row[tag.name].to_numpy()[0]

        red_lo = (
            Maybe(tag.red_bounds)
            .map(lambda bounds: bounds[0])
            .unwrap()
        )
        red_hi = (
            Maybe(tag.red_bounds)
            .map(lambda bounds: bounds[1])
            .unwrap()
        )

        if red_lo is not None and x < red_lo:
            op_lo = (
                Maybe(tag.operating_range)
                .map(lambda rng: rng[0])
                .expect(f'Red zone specified for tag {tag.name}, but no lower bound found')
            )

            return (red_lo - x) / (red_lo - op_lo)

        if red_hi is not None and x > red_hi:
            op_hi = (
                Maybe(tag.operating_range)
                .map(lambda rng: rng[1])
                .expect(f'Red zone specified for tag {tag.name}, but no upper bound found')
            )

            return (x - red_hi) / (op_hi - red_hi)

        return None


def _argmax(arr: list[float | None]) -> int | None:
    v = -1
    idx = None
    for i, x in enumerate(arr):
        if x is None:
            continue

        if x > v:
            v = x
            idx = i

    return idx
