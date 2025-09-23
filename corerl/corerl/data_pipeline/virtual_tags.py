from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.state import AppState
from corerl.tags.components.computed import ComputedTag
from corerl.tags.tag_config import TagConfig
from corerl.utils.sympy import to_sympy


class VirtualTagComputer:
    def __init__(self, tag_cfgs: list[TagConfig], app_state: AppState):
        self._tag_cfgs = tag_cfgs
        self._app_state = app_state
        self._evaluators = {
            tag.name: to_sympy(tag.value)
            for tag in self._tag_cfgs
            if isinstance(tag, ComputedTag)
                and tag.is_computed
                and tag.value is not None
        }

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        for tag, (_, lmda, dependencies) in self._evaluators.items():
            dep_values = [
                pf.data[dep_name].to_numpy()
                for dep_name in dependencies
            ]

            pf.data[tag] = lmda(*dep_values)

        return pf

def log_virtual_tags(
    app_state: AppState,
    prep_stage: Preprocessor,
    tag_cfgs: list[TagConfig],
    pf: PipelineFrame,
):
    """
    Log denormalized delta tags after outliers have been filtered and NaNs have been imputed
    """
    raw_data = prep_stage.inverse(pf.data)
    for tag_cfg in tag_cfgs:
        if isinstance(tag_cfg, ComputedTag) and tag_cfg.is_computed:
            if len(raw_data[tag_cfg.name]) > 0:
                val = float(raw_data[tag_cfg.name].values[-1])
                app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric="VIRTUAL-" + tag_cfg.name,
                    value=val,
                )
