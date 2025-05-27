from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.sympy import to_sympy


class VirtualTagComputer:
    def __init__(self, tag_cfgs: list[TagConfig]):
        self._tag_cfgs = tag_cfgs
        self._evaluators = {
            tag.name: to_sympy(tag.value)
            for tag in self._tag_cfgs
            if tag.is_computed
                and tag.value is not None
        }

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        for tag, (_, lmda, dependencies) in self._evaluators.items():
            dep_values = [
                pf.data[dep_name].to_numpy()
                for dep_name in dependencies
            ]

            pf.data[tag] = lmda(*dep_values)

            # missingness checks come after this stage,
            # so we can prefill the missing_info with NULL here
            # and depend on the "missingness_checker" to do the
            # work for us.
            pf.missing_info[tag] = [MissingType.NULL] * len(pf.data)

        return pf
