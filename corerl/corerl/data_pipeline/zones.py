import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, assert_never

import numpy as np
import pandas as pd
from lib_utils.maybe import Maybe

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.messages.events import RLEvent, RLEventType
from corerl.state import AppState
from corerl.tags.components.bounds import BoundInfo, SafetyZonedTag, ViolationDirection
from corerl.tags.setpoint import eval_bound
from corerl.tags.tag_config import TagConfig

logger = logging.getLogger(__name__)


@dataclass
class ZoneViolation:
    row_idx: Any
    tag: SafetyZonedTag
    percent: float
    direction: ViolationDirection
    kind: Literal['red', 'yellow']


class ZoneDiscourager:
    def __init__(self, app_state: AppState, tag_configs: list[TagConfig], prep_stage: Preprocessor):
        self._app_state = app_state
        self._prep_stage = prep_stage

        self._bounded_tag_cfgs = [
            tag_cfg for tag_cfg in tag_configs
            if isinstance(tag_cfg, SafetyZonedTag)
            and (
                tag_cfg.yellow_bounds is not None
                or tag_cfg.red_bounds is not None
            )
        ]


    def __call__(self, pf: PipelineFrame):
        df = self._prep_stage.inverse(pf.data)

        rewards = pf.rewards['reward'].to_numpy(copy=True)
        for i, (index, row_series) in enumerate(df.iterrows()):
            row = row_series.to_frame().transpose()

            violation, penalty = self._get_penalty_for_row(pf, row, index)
            if violation is None:
                continue

            # red zones are replacing to encode a priority level
            if violation.kind == 'red':
                rewards[i] = penalty

            # yellow zones are additive to encode light penalty
            elif violation.kind == 'yellow':
                rewards[i] += penalty

            # no other types of zones
            else:
                assert_never(violation.kind)


        pf.rewards['reward'] = rewards
        return pf


    def _get_penalty_for_row(self, pf: PipelineFrame, row: pd.DataFrame, row_idx: Any):
        red_violations = [
            self._detect_red_violation(row, row_idx, tag)
            for tag in self._bounded_tag_cfgs
        ]
        max_penalty: float = np.inf
        max_violation: ZoneViolation | None = None
        for violation in red_violations:
            if violation is None:
                continue

            penalty = self._handle_red_violation(pf, violation)
            if penalty < max_penalty:
                max_penalty = penalty
                max_violation = violation


        # then yellow zone
        yellow_violations = [
            self._detect_yellow_violation(row, row_idx, tag)
            for tag in self._bounded_tag_cfgs
        ]
        for violation in yellow_violations:
            if violation is None:
                continue

            penalty = self._handle_yellow_violation(pf, violation)
            if penalty < max_penalty:
                max_penalty = penalty
                max_violation = violation

        return max_violation, max_penalty


    def _handle_yellow_violation(self, pf: PipelineFrame, violation: ZoneViolation):
        """
        Lifecycle method to handle yellow zone violations. Computes the reward penalty
        given degree of violation.

        Reward penalty ramps slowly from 0 to -2.
        """
        if pf.data_mode == DataMode.ONLINE:
            event = ZoneViolationEvent.from_violation(violation)
            self._app_state.event_bus.emit_event(event)
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric='yellow_zone_violation',
                value=violation.percent,
            )
            logger.warning(f"Yellow zone violation for tag {violation.tag.name} at level: {violation.percent}")

        return -2 * (violation.percent**2)


    def _handle_red_violation(
        self,
        pf: PipelineFrame,
        violation: ZoneViolation,
    ):
        """
        Lifecycle method to handle red zone violations. Computes the reward penalty
        given degree of violation.

        Reward penalty ramps quickly from -4 to -8.
        """
        if pf.data_mode == DataMode.ONLINE:
            event = ZoneViolationEvent.from_violation(violation)
            self._app_state.event_bus.emit_event(event)
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric='red_zone_violation',
                value=violation.percent,
            )
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric='yellow_zone_violation',
                value=1.0,
            )
            logger.error(f"Red zone violation for tag {violation.tag.name} at level: {violation.percent}")


        self._apply_red_zone_reaction(pf, violation)
        return -4 - (4 * violation.percent)


    def _detect_yellow_violation(self, row: pd.DataFrame, row_idx: int, tag: SafetyZonedTag):
        x: float = row[tag.name].to_numpy()[0]

        yellow_lo = (
            Maybe[BoundInfo](tag.yellow_bounds_info and tag.yellow_bounds_info.lower)
            .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)
            .unwrap()
        )
        yellow_hi = (
            Maybe[BoundInfo](tag.yellow_bounds_info and tag.yellow_bounds_info.upper)
            .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)
            .unwrap()
        )

        if yellow_lo is not None and x < yellow_lo:
            next_lo = (
                # the next lowest bound is either the red zone if one exists
                Maybe[BoundInfo](tag.red_bounds_info and tag.red_bounds_info.lower)
                .map(partial(eval_bound, row))

                # or the operating bound if one exists
                .otherwise(lambda: tag.operating_bounds_info and tag.operating_bounds_info.lower)
                .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)

                # and if neither exists, we're in trouble
                .expect(f'Yellow zone specified for tag {tag.name}, but no lower bound found')
            )
            return ZoneViolation(
                row_idx=row_idx,
                tag=tag,
                percent=np.clip((yellow_lo - x) / (yellow_lo - next_lo), 0, 1),
                direction=ViolationDirection.lower_violation,
                kind='yellow',
            )

        if yellow_hi is not None and x > yellow_hi:
            next_hi = (
                # the next highest bound is either the red zone if one exists
                Maybe[BoundInfo](tag.red_bounds_info and tag.red_bounds_info.upper)
                .map(partial(eval_bound, row))

                # or the operating bound if one exists
                .otherwise(lambda: tag.operating_bounds_info and tag.operating_bounds_info.upper)
                .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)

                # and if neither exists, we're in trouble
                .expect(f'Yellow zone specified for tag {tag.name}, but no upper bound found')
            )
            return ZoneViolation(
                row_idx=row_idx,
                tag=tag,
                percent=np.clip((x - yellow_hi) / (next_hi - yellow_hi), 0, 1),
                direction=ViolationDirection.upper_violation,
                kind='yellow',
            )

        return None

    def _detect_red_violation(self, row: pd.DataFrame, row_idx: Any, tag: SafetyZonedTag):
        x: float = row[tag.name].to_numpy()[0]

        red_lo = (
            Maybe[BoundInfo](tag.red_bounds_info and tag.red_bounds_info.lower)
            .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)
            .unwrap()
        )
        red_hi = (
            Maybe[BoundInfo](tag.red_bounds_info and tag.red_bounds_info.upper)
            .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)
            .unwrap()
        )

        if red_lo is not None and x < red_lo:
            op_lo = (
                Maybe[BoundInfo](tag.operating_bounds_info and tag.operating_bounds_info.lower)
                .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)
                .expect(f'Red zone specified for tag {tag.name}, but no lower bound found')
            )
            return ZoneViolation(
                row_idx=row_idx,
                tag=tag,
                percent=(red_lo - x) / (red_lo - op_lo),
                direction=ViolationDirection.lower_violation,
                kind='red',
            )

        if red_hi is not None and x > red_hi:
            op_hi = (
                Maybe[BoundInfo](tag.operating_bounds_info and tag.operating_bounds_info.upper)
                .map(partial(eval_bound, row)).map(lambda x: x.float_bound).is_instance(float)
                .expect(f'Red zone specified for tag {tag.name}, but no upper bound found')
            )
            return ZoneViolation(
                row_idx=row_idx,
                tag=tag,
                percent=(x - red_hi) / (op_hi - red_hi),
                direction=ViolationDirection.upper_violation,
                kind='red',
            )

        return None


    def _apply_red_zone_reaction(self, pf: PipelineFrame, violation: ZoneViolation):
        if violation.tag.red_zone_reaction is None:
            return

        for reflex_cfg in violation.tag.red_zone_reaction:
            if violation.direction != reflex_cfg.kind:
                continue

            lo, hi = reflex_cfg.bounds
            if lo is not None:
                pf.action_lo.loc[violation.row_idx, f'{reflex_cfg.tag}-lo'] = lo

            if hi is not None:
                pf.action_hi.loc[violation.row_idx, f'{reflex_cfg.tag}-hi'] = hi


class ZoneViolationEvent(RLEvent):
    tag: str
    percent: float
    direction: ViolationDirection

    @staticmethod
    def from_violation(violation: ZoneViolation):
        return ZoneViolationEvent(
            type=RLEventType.red_zone_violation if violation.kind == 'red' else RLEventType.yellow_zone_violation,
            tag=violation.tag.name,
            percent=violation.percent,
            direction=violation.direction,
        )
