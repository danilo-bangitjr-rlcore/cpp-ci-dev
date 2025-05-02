from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, outlier_group
from corerl.data_pipeline.oddity_filters.config import GlobalOddityFilterConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.state import AppState


class OddityFilterConstructor:
    def __init__(self, tag_cfgs: list[TagConfig], app_state: AppState, cfg: GlobalOddityFilterConfig):
        self._app_state = app_state
        self._cfg = cfg
        self._relevant_cfgs = self._get_relevant_configs(tag_cfgs)

        self._components: dict[str, list[BaseOddityFilter]] = {
            tag_name: self._construct_components(filter)
            for tag_name, filter in self._relevant_cfgs.items()
            if filter is not None
        }

        self._tag_cfgs = {tag.name: tag for tag in tag_cfgs}

    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]) -> dict[str, list[OddityFilterConfig]]:
        return {
            tag.name: tag.outlier if tag.outlier is not None else self._cfg.defaults
            for tag in tag_cfgs
            if tag.type != TagType.meta
        }

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        for tag_name, filters in self._components.items():
            for f in filters:
                pf = f(pf, tag_name)
        return pf

    def _construct_components(self, sub_cfgs: list[OddityFilterConfig]):
        return [outlier_group.dispatch(sub_cfg, self._app_state) for sub_cfg in sub_cfgs]
