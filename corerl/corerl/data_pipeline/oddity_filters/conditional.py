import logging

import numpy as np
from lib_utils.maybe import Maybe

from corerl.configs.data_pipeline.oddity_filters.conditional import ConditionalFilterConfig
from corerl.configs.tags.components.opc import OPCTag
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, outlier_group
from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.state import AppState

logger = logging.getLogger(__name__)


class ConditionalFilter(BaseOddityFilter):
    """
    A conditional filter that sets rows to NaN when a condition evaluates to True.
    """
    def __init__(self, cfg: ConditionalFilterConfig, app_state: AppState):
        super().__init__(cfg, app_state)
        self.condition_transforms = [transform_group.dispatch(transform_cfg) for transform_cfg in cfg.condition]
        self.filtered_tags = cfg.filtered_tags
        self.excluded_tags = cfg.excluded_tags

    def __call__(self, pf: PipelineFrame, tag: str, ts: object | None, update_stats: bool = True):
        tag_cfg = (
            Maybe.find(lambda cfg: cfg.name == tag, self._app_state.cfg.pipeline.tags)
            .is_instance(OPCTag)
            .unwrap()
        )

        # Only filter OPC tags in self.filtered_tags or that aren't in self.excluded tags
        if (tag_cfg is None or
            tag in self.excluded_tags or
            (self.filtered_tags != 'all' and tag not in self.filtered_tags)):
            return pf, ts

        # Preparing data to be passed to the list of transforms in the condition
        tag_data = pf.data.get([tag], None)
        assert tag_data is not None

        carry = TransformCarry(
            obs=pf.data,
            transform_data=tag_data.copy(),
            tag=tag,
        )

        # Typing magic to ensure the second argument passed to transform() is an object | None
        sub_ts: list[object | None]
        if ts is not None and isinstance(ts, list):
            sub_ts = ts
        else:
            sub_ts = [None] * len(self.condition_transforms)

        for i, transform in enumerate(self.condition_transforms):
            carry, sub_ts[i] = transform(carry, sub_ts[i])

        # Get the boolean mask output by the transforms and filter out the rows where the mask is True
        result = carry.transform_data
        n_cols = len(result.columns)
        # Ensure the boolean mask is a single column
        if n_cols > 1:
            logger.warning(f"Conditional filter for {tag} resulted in {n_cols} columns")
            return pf, sub_ts
        if n_cols < 1:
            return pf, sub_ts
        try:
            filter_mask = _to_mask(cond=result.iloc[:,0].to_numpy())
        except Exception:
            logger.warning(f"Conditional filter for {tag} could not be cast to bool")
            return pf, sub_ts

        pf.data.loc[filter_mask, tag] = np.nan

        return pf, sub_ts

outlier_group.dispatcher(ConditionalFilter)


def _to_mask(cond: np.ndarray) -> np.ndarray:
    """Convert condition values to boolean mask."""
    mask = np.empty_like(cond, dtype=bool)
    for i, x in enumerate(cond):
        if np.isnan(x):
            mask[i] = False
            continue
        mask[i] = bool(x)
    return mask
