from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig, get_action_bounds
from corerl.utils.maybe import Maybe


class ActionConstructor:
    def __init__(self, tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
        self.action_tags = [tag for tag in tag_cfgs if tag.action_constructor is not None]

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
                lo, hi = get_action_bounds(action_tag, row)
                operating_range = Maybe(action_tag.operating_range).expect()

                lo_val = max(Maybe(operating_range[0]).expect(), lo)
                hi_val = min(Maybe(operating_range[1]).expect(), hi)

                a_lo[action_tag.name] = self._prep_stage.normalize(action_tag.name, lo_val)
                a_hi[action_tag.name] = self._prep_stage.normalize(action_tag.name, hi_val)

            a_los.append(a_lo)
            a_his.append(a_hi)

        pf.actions = pf.data.loc[:, self.columns] # self.columns is sorted
        pf.action_lo = pd.DataFrame(a_los, index=pf.data.index).loc[:, self.columns]
        pf.action_hi = pd.DataFrame(a_his, index=pf.data.index).loc[:, self.columns]
        breakpoint()

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

    def sort_cols(self, cols: Iterable[str]):
        return sorted(cols)
