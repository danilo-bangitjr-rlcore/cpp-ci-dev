import pandas as pd

# ensure components are registered
import corerl.data_pipeline.state_constructors.components.add_raw  # noqa: F401
import corerl.data_pipeline.state_constructors.components.affine  # noqa: F401
import corerl.data_pipeline.state_constructors.components.identity  # noqa: F401
import corerl.data_pipeline.state_constructors.components.norm  # noqa: F401
import corerl.data_pipeline.state_constructors.components.null  # noqa: F401
import corerl.data_pipeline.state_constructors.components.scale  # noqa: F401
import corerl.data_pipeline.state_constructors.components.split  # noqa: F401
import corerl.data_pipeline.state_constructors.components.trace  # noqa: F401
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, TagName
from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig, Transform, sc_group
from corerl.data_pipeline.state_constructors.interface import TransformCarry
from corerl.data_pipeline.utils import invoke_stage_per_tag

type RC_TS = dict[
    # tag name
    str,
    # transform steps
    list[object | None],
]


class RewardComponentConstructor:
    def __init__(self, cfgs: list[BaseTransformConfig]):
        self._components: list[Transform] = [sc_group.dispatch(sub_cfg) for sub_cfg in cfgs]

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

        for i in range(len(self._components)):
            transform = self._components[i]
            transform_ts = tag_ts[i]

            carry, transform_ts = transform(carry, transform_ts)
            tag_ts[i] = transform_ts

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_name, axis=1, inplace=False)
        carry.transform_data = carry.transform_data.rename(columns=lambda x: "(reward)" + x)
        pf.data = pd.concat((df, carry.transform_data), axis=1, copy=False)

        # put new temporal state on PipeFrame
        pf.temporal_state[StageCode.SC] = ts
        return pf

    def _sanitize_temporal_state(self, ts: object | None, tag_name: str):
        if ts is None:
            ts = {}

        assert isinstance(ts, dict)
        if tag_name not in ts:
            ts[tag_name] = [None] * len(self._components)

        return ts[tag_name]


class RewardConstructor:
    def __init__(self, component_constructors: dict[TagName, RewardComponentConstructor]):
        self.component_constructors = component_constructors

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        initial_data = pf.data.copy(deep=True)
        pf = invoke_stage_per_tag(pf, self.component_constructors)
        reward_component_names = [col for col in pf.data.columns if "(reward)" in col]
        reward_components = pf.data[reward_component_names]
        # add final reward column
        if not reward_components.empty:
            initial_data["reward"] = reward_components.sum(axis=1, skipna=False)

        pf.data = initial_data

        return pf
