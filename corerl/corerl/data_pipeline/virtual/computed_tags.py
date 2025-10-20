from collections.abc import Sequence
from typing import Protocol

from corerl.configs.tags.tag_config import TagConfig
from corerl.data_pipeline.datatypes import PipelineFrame


class ComputedTagComputer(Protocol):
    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        """
        Compute and add tags to the PipelineFrame.
        """
        ...

    def reset(self):
        """
        Reset any stateful components. Optional - not all computers need state.
        """
        ...


class ComputedTagStage:
    def __init__(self, tag_cfgs: Sequence[TagConfig], *computers: ComputedTagComputer):
        self.tag_cfgs = tag_cfgs
        self.computers = computers

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        for computer in self.computers:
            pf = computer(pf)
        return pf

    def reset(self):
        for computer in self.computers:
            computer.reset()
