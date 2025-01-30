from collections.abc import Sequence

import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, TagName
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.base import Transform, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.null import Null
from corerl.data_pipeline.utils import get_tag_temporal_state, invoke_stage_per_tag


class RewardComponentConstructor:
    def __init__(self, cfgs: Sequence[TransformConfig]):
        self._transforms: list[Transform] = [transform_group.dispatch(sub_cfg) for sub_cfg in cfgs]

    def __call__(self, pf: PipelineFrame, tag_name: str) -> PipelineFrame:
        tag_data = pf.rewards.get([tag_name])
        assert tag_data is not None

        carry = TransformCarry(
            obs=pf.rewards,
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
        pf.rewards = pd.concat((pf.rewards, carry.transform_data), axis=1, copy=False)

        return pf


class RewardConstructor:
    def __init__(self, tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
        self.component_constructors = _filter_null_components(tag_cfgs)
        self._tag_cfgs = tag_cfgs
        self._prep_stage = prep_stage

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        # denormalize all tags before invoking stages per tag
        # this way any xforms that depend on "other" will get the
        # denormalized version of other
        pf.rewards = self.denormalize_tags(pf.data)

        # perform temp xforms for all tags
        pf = invoke_stage_per_tag(pf, self.component_constructors)

        # then merge their results together
        reward_component_names = [col for col in pf.rewards.columns if "[reward]" in col]
        reward_components = pf.rewards[reward_component_names]
        # add final reward column
        if not reward_components.empty:
            reward = reward_components.sum(axis=1, skipna=False)
            pf.rewards = pd.DataFrame({
                'reward': reward,
            })

        return pf


    def denormalize_tags(self, df: pd.DataFrame):
        return self._prep_stage.inverse(df)


def _filter_null_components(
    tag_cfgs: list[TagConfig],
) -> dict[TagName, RewardComponentConstructor]:
    nonnull_ccs = {}
    for tag in tag_cfgs:
        xforms = tag.reward_constructor
        null_cc = len(xforms) == 1 and isinstance(xforms[0], Null)
        if not null_cc:
            nonnull_ccs[tag.name] = RewardComponentConstructor(tag.reward_constructor)

    return nonnull_ccs
