from collections.abc import Iterable
from datetime import timedelta
from functools import cached_property

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig, TagType, get_action_bounds
from corerl.state import AppState
from corerl.utils.maybe import Maybe
from corerl.utils.time import percent_time_elapsed


class ActionConstructor:
    def __init__(self, app_state: AppState, tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
        self._app_state = app_state
        self.action_tags = [tag for tag in tag_cfgs if tag.type == TagType.ai_setpoint]

        # make sure operating ranges are specified for actions
        for action_tag in self.action_tags:
            name = action_tag.name
            Maybe(action_tag.operating_range).map(lambda r: r[0]).expect(
                f"Action {name} did not specify an operating range lower bound."
            )
            Maybe(action_tag.operating_range).map(lambda r: r[1]).expect(
                f"Action {name} did not specify an operating range upper bound."
            )
        self._prep_stage = prep_stage # used to denormalize tags


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        # denormalize all tags before computing action bounds
        raw_data = self.denormalize_tags(pf.data)

        a_los: list[dict[str, float]] = []
        a_his: list[dict[str, float]] = []
        for _, row_series in raw_data.iterrows():
            """
            Generate one dict each for action lower bounds and upper bounds
            """
            row = row_series.to_frame().transpose()
            a_lo = {}
            a_hi = {}
            for action_tag in self.action_tags:
                ab_lo, ab_hi = get_action_bounds(action_tag, row)
                operating_range = Maybe(action_tag.operating_range).expect()
                op_lo, op_hi = Maybe(operating_range[0]).expect(), Maybe(operating_range[1]).expect()
                ab_lo = max(op_lo, ab_lo)
                ab_hi = min(op_hi, ab_hi)

                # overwrite the action bounds/operating range with the guardrails if they exist
                maybe_guard = self._get_guardrails(action_tag, op_lo, op_hi)
                maybe_guard_lo = Maybe(maybe_guard[0])
                maybe_guard_hi = Maybe(maybe_guard[1])

                # Apply the guardrails if they exist
                ab_lo = max(maybe_guard_lo.or_else(ab_lo), ab_lo)
                ab_hi = min(maybe_guard_hi.or_else(ab_hi), ab_hi)

                # normalize the action bounds to the operating range
                a_lo[f"{action_tag.name}-lo"] = self._prep_stage.normalize(action_tag.name, ab_lo)
                a_hi[f"{action_tag.name}-hi"] = self._prep_stage.normalize(action_tag.name, ab_hi)

            a_los.append(a_lo)
            a_his.append(a_hi)

        pf.actions = pf.data.loc[:, self.columns] # self.columns is sorted
        pf.action_lo = pd.DataFrame(a_los, index=pf.data.index).loc[:, self.action_lo_columns]
        pf.action_hi = pd.DataFrame(a_his, index=pf.data.index).loc[:, self.action_hi_columns]

        return pf

    def get_action_df(self, action_arr: np.ndarray) -> pd.DataFrame:
        """
        Given an action, return a dataframe that contains the action info and the corresponding column names
        """
        df = pd.DataFrame(data=[action_arr], columns=self.columns)

        return df

    def denormalize_tags(self, df: pd.DataFrame):
        return self._prep_stage.inverse(df)

    @cached_property
    def columns(self):
        return self.sort_cols([tag.name for tag in self.action_tags])

    @cached_property
    def action_lo_columns(self):
        return self.sort_cols([f"{tag.name}-lo" for tag in self.action_tags])

    @cached_property
    def action_hi_columns(self):
        return self.sort_cols([f"{tag.name}-hi" for tag in self.action_tags])

    def sort_cols(self, cols: Iterable[str]):
        return sorted(cols)

    def _get_guardrails(self, cfg: TagConfig, a_lo: float, a_hi: float):
        guard_lo, guard_hi = None, None

        if cfg.guardrail_schedule is None:
            return guard_lo, guard_hi

        perc = self._get_elapsed_guardrail_duration(cfg.guardrail_schedule.duration)

        # if we are at the end of the guardrail duration, return None to signal to over the operating range in call()
        if perc > 1:
            return guard_lo, guard_hi

        start_lo, start_hi = cfg.guardrail_schedule.starting_range

        if start_lo is not None:
            guard_lo = (1 - perc) * start_lo + perc * a_lo

        if start_hi is not None:
            guard_hi = (1 - perc) * start_hi + perc * a_hi

        return guard_lo, guard_hi

    def _get_elapsed_guardrail_duration(self, guardrail_duration: timedelta):
        return percent_time_elapsed(
            start=self._app_state.start_time,
            end=self._app_state.start_time + guardrail_duration,
        )
