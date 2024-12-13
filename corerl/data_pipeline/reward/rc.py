import pandas as pd

# ensure components are registered
import corerl.data_pipeline.transforms.add_raw  # noqa: F401
import corerl.data_pipeline.transforms.affine  # noqa: F401
import corerl.data_pipeline.transforms.identity  # noqa: F401
import corerl.data_pipeline.transforms.norm  # noqa: F401
import corerl.data_pipeline.transforms.null  # noqa: F401
import corerl.data_pipeline.transforms.scale  # noqa: F401
import corerl.data_pipeline.transforms.split  # noqa: F401
import corerl.data_pipeline.transforms.trace  # noqa: F401
from corerl.data_pipeline.transforms.null import Null
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, TagName
from corerl.data_pipeline.transforms.base import BaseTransformConfig, Transform, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.utils import invoke_stage_per_tag

type RC_TS = dict[
    # tag name
    str,
    # transform steps
    list[object | None],
]


class RewardComponentConstructor:
    def __init__(self, cfgs: list[BaseTransformConfig]):
        self._transforms: list[Transform] = [transform_group.dispatch(sub_cfg) for sub_cfg in cfgs]

    def __call__(self, pf: PipelineFrame, tag_name: str) -> PipelineFrame:
        tag_data = pf.data.get([tag_name])
        assert tag_data is not None

        carry = TransformCarry(
            obs=pf.data,
            transform_data=tag_data.copy(),
            tag=tag_name,
        )

        ts = pf.temporal_state.get(StageCode.RC, None)
        tag_ts = self._sanitize_temporal_state(ts, tag_name)

        for i in range(len(self._transforms)):
            transform = self._transforms[i]
            transform_ts = tag_ts[i]

            carry, transform_ts = transform(carry, transform_ts)
            tag_ts[i] = transform_ts

        # put resultant data on PipeFrame
        carry.transform_data = carry.transform_data.rename(columns=lambda x: "[reward]" + x)
        pf.data = pd.concat((pf.data, carry.transform_data), axis=1, copy=False)

        # put new temporal state on PipeFrame
        pf.temporal_state[StageCode.RC] = ts
        return pf

    def _sanitize_temporal_state(self, ts: object | None, tag_name: str):
        if ts is None:
            ts = {}

        assert isinstance(ts, dict)
        if tag_name not in ts:
            ts[tag_name] = [None] * len(self._transforms)

        return ts[tag_name]


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
