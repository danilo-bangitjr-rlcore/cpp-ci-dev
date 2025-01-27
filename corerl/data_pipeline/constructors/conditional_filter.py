import logging
from functools import cached_property

import numpy as np

from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import MissingType, PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.utils import update_missing_info

logger = logging.getLogger(__name__)


class ConditionalFilter(Constructor):
    def __init__(self, tag_cfgs: list[TagConfig]):
        super().__init__(tag_cfgs)

    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]):
        return {
            tag.name: tag.filter
            for tag in tag_cfgs
            if tag.filter is not None and not tag.is_meta
        }

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        transformed_parts, tag_names = self._transform_tags(pf, StageCode.PREPROCESS)
        for tag, result in zip(tag_names, transformed_parts, strict=True):
            n_cols = len(result.columns)
            if n_cols > 1:
                logger.warning(f"Conditional filter for {tag} resulted in {n_cols} columns")
                continue
            if n_cols < 1:
                continue
            try:
                filter_mask = _to_mask(cond=result.iloc[:,0].to_numpy())
            except Exception:
                logger.warning(f"Conditional filter for {tag} could not be cast to bool")
                continue
            pf.data.loc[filter_mask, tag] = np.nan
            update_missing_info(pf.missing_info, tag, filter_mask, MissingType.FILTER)

        return pf

    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return list(pf.data.columns)

def _to_mask(cond: np.ndarray) -> np.ndarray:
    mask = np.empty_like(cond, dtype=bool)
    for i, x in enumerate(cond):
        if np.isnan(x):
            mask[i] = False
            continue
        mask[i] = bool(x)
    return mask
