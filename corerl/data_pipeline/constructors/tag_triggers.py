import logging
from functools import cached_property

import numpy as np

from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig
from corerl.messages.events import EventTopic
from corerl.state import AppState

logger = logging.getLogger(__name__)


class TagTrigger(Constructor):
    def __init__(self, app_state: AppState, tag_cfgs: list[TagConfig]):
        super().__init__(tag_cfgs)
        self._app_state = app_state

    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]):
        return {
            tag.name: tag.trigger.condition
            for tag in tag_cfgs
            if tag.trigger is not None
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if pf.data_mode != DataMode.ONLINE:
            return pf

        transformed_parts, tag_names = self._transform_tags(pf, StageCode.TRIGGER)
        for tag, result in zip(tag_names, transformed_parts, strict=True):
            n_cols = len(result.columns)
            if n_cols > 1:
                logger.warning(f"Tag trigger for {tag} resulted in {n_cols} columns")
                continue
            if n_cols < 1:
                continue
            try:
                filter_mask = _to_mask(cond=result.iloc[:,0].to_numpy())
            except Exception:
                logger.warning(f"Tag trigger for {tag} could not be cast to bool")
                continue

            if filter_mask.any():
                trigger_cfg = self._tag_cfgs[tag].trigger
                assert trigger_cfg is not None
                self._app_state.event_bus.emit_event(
                    trigger_cfg.event,
                    topic=EventTopic.corerl,
                )

        return pf

    @cached_property
    def columns(self):
        return []

def _to_mask(cond: np.ndarray) -> np.ndarray:
    mask = np.empty_like(cond, dtype=bool)
    for i, x in enumerate(cond):
        if np.isnan(x):
            mask[i] = False
            continue
        mask[i] = bool(x)
    return mask
