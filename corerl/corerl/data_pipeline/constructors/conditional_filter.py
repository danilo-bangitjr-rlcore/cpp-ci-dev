import logging
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from lib_defs.config_defs.tag_config import TagType

from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.tags.tag_config import TagConfig

logger = logging.getLogger(__name__)


class ConditionalFilter(Constructor):
    def __init__(self, tag_cfgs: Sequence[TagConfig]):
        super().__init__(tag_cfgs)

    def _get_relevant_configs(self, tag_cfgs: Sequence[TagConfig]):
        return {
            tag.name: tag.filter
            for tag in tag_cfgs
            if tag.filter is not None and tag.type != TagType.meta
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
