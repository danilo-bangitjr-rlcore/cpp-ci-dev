from functools import cached_property

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.maybe import Maybe


class ActionConstructor(Constructor):
    def __init__(self, tag_cfgs: list[TagConfig]):
        super().__init__(tag_cfgs)
        # make sure operating ranges are specified for actions
        for name, tag_cfg in self._tag_cfgs.items():
            if tag_cfg.action_constructor is None:
                continue
            Maybe(tag_cfg.operating_range).map(lambda r: r[0]).expect(
                f"Action {name} did not specify an operating range lower bound."
            )
            Maybe(tag_cfg.operating_range).map(lambda r: r[1]).expect(
                f"Action {name} did not specify an operating range upper bound."
            )

    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]):
        return {
            tag.name: tag.action_constructor
            for tag in tag_cfgs
            if not tag.is_meta and tag.action_constructor is not None
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        transformed_parts, _ = self._transform_tags(pf, StageCode.AC)

        if len(transformed_parts) == 0:
            return pf

        # put resultant data on PipeFrame
        pf.actions = pd.concat(transformed_parts, axis=1, copy=False)

        # guarantee an ordering over columns
        sorted_cols = sorted(pf.actions.columns)
        pf.actions = pf.actions.loc[:, sorted_cols]

        return pf


    def assign_action_names(self, action_arr: np.ndarray):
        """
        Because the action constructor is responsible for setting action ordering,
        then when we receive a numpy array with a magic ordering, the AC is
        responsible for labelling those ordered actions with their respective names.
        """
        d = {col: [act] for col, act in zip(self.columns, action_arr, strict=True)}
        df = pd.DataFrame(d)
        return df


    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return sorted(pf.actions.columns)
