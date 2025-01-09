import pandas as pd

from collections.abc import Sequence
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.null import Null
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, TagName
from corerl.data_pipeline.transforms.base import Transform, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.utils import get_tag_temporal_state, invoke_stage_per_tag


class RewardComponentConstructor:
    def __init__(self, cfgs: Sequence[TransformConfig]):
        self._transforms: list[Transform] = [transform_group.dispatch(sub_cfg) for sub_cfg in cfgs]

    def __call__(self, pf: PipelineFrame, tag_name: str) -> PipelineFrame:
        tag_data = pf.data.get([tag_name])
        assert tag_data is not None

        carry = TransformCarry(
            obs=pf.data,
            transform_data=tag_data.copy(),
            tag=tag_name,
        )

        tag_ts: list[object | None] = get_tag_temporal_state(
            StageCode.RC, tag_name, pf.temporal_state,
            default=lambda: [None] * len(self._transforms),
        )

        for i in range(len(self._transforms)):
            transform = self._transforms[i]
            transform_ts = tag_ts[i]

            carry, transform_ts = transform(carry, transform_ts)
            tag_ts[i] = transform_ts

        # put resultant data on PipeFrame
        carry.transform_data = carry.transform_data.rename(columns=lambda x: "[reward]" + x)
        pf.data = pd.concat((pf.data, carry.transform_data), axis=1, copy=False)

        return pf

    def reset(self) -> None:
        for transform in self._transforms:
            transform.reset()


class RewardConstructor:
    def __init__(self, component_constructors: dict[TagName, RewardComponentConstructor]):
        self.component_constructors = _filter_null_components(component_constructors)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        initial_data = pf.data.copy(deep=True)
        pf = invoke_stage_per_tag(pf, self.component_constructors)
        reward_component_names = [col for col in pf.data.columns if "[reward]" in col]
        reward_components = pf.data[reward_component_names]
        # add final reward column
        if not reward_components.empty:
            initial_data["reward"] = reward_components.sum(axis=1, skipna=False)

        pf.data = initial_data

        return pf

    def reset(self) -> None:
        for component in self.component_constructors.values():
            component.reset()


def _filter_null_components(
    component_constructors: dict[TagName, RewardComponentConstructor],
) -> dict[TagName, RewardComponentConstructor]:
    nonnull_ccs = {}
    for tagname, cc in component_constructors.items():
        xforms = cc._transforms
        null_cc = len(xforms) == 1 and isinstance(xforms[0], Null)
        if not null_cc:
            nonnull_ccs[tagname] = cc

    return nonnull_ccs
