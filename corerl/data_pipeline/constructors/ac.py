from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms.base import InvertibleTransform
from corerl.data_pipeline.transforms.delta import Delta
from corerl.utils.list import find, find_index
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
        sorted_cols = self.sort_cols(pf.actions.columns)
        pf.actions = pf.actions.loc[:, sorted_cols]

        return pf


    def assign_action_names(self, offset_arr: np.ndarray, delta_arr: np.ndarray):
        """
        Because the action constructor is responsible for setting action ordering,
        then when we receive a numpy array with a magic ordering, the AC is
        responsible for labelling those ordered actions with their respective names.
        """
        actions: dict[str, float] = {}
        for action_idx, tag_name in enumerate(self._relevant_cfgs.keys()):
            delta_idx = find_index(
                lambda c: c.startswith(tag_name) and Delta.is_delta_transformed(c), # noqa: B023
                self.columns,
            )

            # if there is no delta action
            # then there is no offset. So the delta_arr
            # actually just contains the action itself
            if delta_idx is None:
                actions[tag_name] = delta_arr[action_idx]
                continue

            # otherwise, we need to find the direct action
            # in the offset list, and add these together
            direct_idx = find_index(
                lambda c: c.startswith(tag_name) and not Delta.is_delta_transformed(c), # noqa: B023
                self.columns,
            )
            assert direct_idx is not None, 'failed to find a direct action index when a delta action index exists'

            inverted_delta = self.invert(
                np.array([delta_arr[action_idx]]),
                self.columns[delta_idx],
            )

            # because we are operating in normalized action space
            # we know that the bounds are strictly [0, 1] here
            actions[tag_name] = np.clip(offset_arr[direct_idx] + inverted_delta, 0, 1)

        d = {col: [act] for col, act in actions.items()}
        df = pd.DataFrame(d)
        return df


    def invert(self, action: np.ndarray, col: str):
        tag_name = find(lambda tag: col.startswith(tag), self._components)
        assert tag_name is not None, f"Could not find AC xforms for col {col}"

        for xform in reversed(self._components[tag_name]):
            if isinstance(xform, InvertibleTransform):
                action = xform.invert(action, col)

        return action


    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return self.sort_cols(pf.actions.columns)


    def sort_cols(self, cols: Iterable[str]):
        return sorted(cols)
