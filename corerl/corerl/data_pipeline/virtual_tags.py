from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
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

            if len(pf.data[tag]) > 0:
                val = float(pf.data[tag].values[0])
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric="VIRTUAL-" + tag,
                    value=val,
                )

            # missingness checks come after this stage,
            # so we can prefill the missing_info with NULL here
            # and depend on the "missingness_checker" to do the
            # work for us.
            pf.missing_info[tag] = [MissingType.NULL] * len(pf.data)

        return pf
