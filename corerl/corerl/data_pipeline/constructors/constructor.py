from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property

import pandas as pd

from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.base import Transform, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


class Constructor(ABC):
    def __init__(self, tag_cfgs: list[TagConfig]):
        self._relevant_cfgs = self._get_relevant_configs(tag_cfgs)

        self._components: dict[str, list[Transform]] = {
            tag_name: self._construct_components(transforms)
            for tag_name, transforms in self._relevant_cfgs.items()
            if transforms is not None
        }

        self._tag_cfgs = {
            tag.name: tag
            for tag in tag_cfgs
        }

    # ------------------------
    # -- Required Overrides --
    # ------------------------
    @abstractmethod
    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]) -> dict[str, list[TransformConfig]]:
        ...


    @abstractmethod
    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ...


    @cached_property
    @abstractmethod
    def columns(self) -> list[str]:
        ...


    # ----------------
    # -- Public API --
    # ----------------
    def reset(self) -> None:
        for transforms in self._components.values():
            for transform in transforms:
                transform.reset()


    # ------------------------
    # -- Internal Utilities --
    # ------------------------
    def _probe_fake_data(self):
        tag_names = self._components.keys()

        fake_data = pd.DataFrame({
            tag_name: [1., 0.]
            for tag_name in tag_names
        })
        fake_indices = pd.DatetimeIndex(["7/13/2023 10:00", "7/13/2023 11:00"])
        fake_data.index = fake_indices

        pf = PipelineFrame(
            data=fake_data,
            data_mode=DataMode.OFFLINE,
            temporal_state=defaultdict(lambda: None),
        )

        action_tags = [self._tag_cfgs[tag_name] for tag_name in self._tag_cfgs if
                       self._tag_cfgs[tag_name].type == TagType.ai_setpoint]
        action_los = pd.DataFrame({
            f"{action_tag.name}-lo": [0., 0.]
            for action_tag in action_tags
        })
        action_los.index = fake_indices

        action_his = pd.DataFrame({
            f"{action_tag.name}-hi": [1., 1.]
            for action_tag in action_tags
        })
        action_his.index = fake_indices

        pf.action_lo = action_los
        pf.action_hi = action_his

        pf = self(pf)

        # ensure that the dummy does not mutate any
        # transform states
        self.reset()
        return pf


    def _transform_tags(
        self,
        pf: PipelineFrame,
        stage_code: StageCode,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        ts = pf.temporal_state.get(stage_code, {})
        assert isinstance(ts, dict)

        tag_names = list(self._components.keys())
        if len(tag_names) == 0:
            return [], tag_names

        transformed_parts = [
            self._invoke_per_tag(pf.data, tag_name, ts)
            for tag_name in tag_names
        ]

        # put new temporal state on PipeFrame
        pf.temporal_state[stage_code] = ts

        return transformed_parts, tag_names


    def _invoke_per_tag(self, df: pd.DataFrame, tag_name: str, ts: dict[str, list[object | None]]) -> pd.DataFrame:
        tag_data = df.get([tag_name], None)
        assert tag_data is not None

        carry = TransformCarry(
            obs=df,
            transform_data=tag_data.copy(),
            tag=tag_name,
        )

        transforms = self._components[tag_name]

        # make a default ts if one doesn't already exist
        # and attach it back to the shared ts
        sub_ts = ts.get(tag_name, [None] * len(transforms))
        ts[tag_name] = sub_ts

        for i, transform in enumerate(transforms):
            carry, sub_ts[i] = transform(carry, sub_ts[i])

        return carry.transform_data


    def _construct_components(self, sub_cfgs: list[TransformConfig]):
        return [
            transform_group.dispatch(sub_cfg) for sub_cfg in sub_cfgs
        ]
