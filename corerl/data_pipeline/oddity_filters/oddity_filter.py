from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
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
        ts = pf.temporal_state.get(StageCode.ODDITY, {})
        assert isinstance(ts, dict)

        tag_names = list(self._components.keys())
        if len(tag_names) == 0:
            return pf

        for tag_name in tag_names:
            pf = self._invoke_per_tag(pf, tag_name, ts)

        # put new temporal state on PipeFrame
        pf.temporal_state[StageCode.ODDITY] = ts

        return pf

    def _invoke_per_tag(self, pf: PipelineFrame, tag_name: str, ts: dict[str, list[object | None]]) -> PipelineFrame:

        filters = self._components[tag_name]

        # make a default ts if one doesn't already exist
        # and attach it back to the shared ts
        sub_ts = ts.get(tag_name, [None] * len(filters))
        ts[tag_name] = sub_ts

        for i, f in enumerate(filters):
            carry, sub_ts[i] = f(pf, tag_name, sub_ts[i])

        return pf

    def _construct_components(self, sub_cfgs: list[OddityFilterConfig]):
        return [outlier_group.dispatch(sub_cfg, self._app_state) for sub_cfg in sub_cfgs]
