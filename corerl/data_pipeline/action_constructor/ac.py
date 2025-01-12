import pandas as pd

from collections import defaultdict
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.transforms.base import transform_group, Transform
from corerl.data_pipeline.tag_config import TagConfig


class ActionConstructor:
    def __init__(self, tag_cfgs: list[TagConfig]):
        cfgs = {
            tag.name: tag.action_constructor
            for tag in tag_cfgs
            if not tag.is_meta
        }

        self._components: dict[str, list[Transform]] = {
            tag_name: self._construct_components(parts)
            for tag_name, parts in cfgs.items()
            if parts is not None
        }


    def _construct_components(self, sub_cfgs: list[TransformConfig]):
        return [
            transform_group.dispatch(sub_cfg) for sub_cfg in sub_cfgs
        ]


    def action_columns(self):
        tag_names = self._components.keys()

        fake_data = pd.DataFrame({
            tag_name: [1., 0.]
            for tag_name in tag_names
        })

        pf = PipelineFrame(
            data=fake_data,
            caller_code=CallerCode.OFFLINE,
            temporal_state=defaultdict(lambda: None),
        )

        pf = self(pf)

        # ensure that the dummy does not mutate any
        # transform states
        self.reset()

        return list(pf.actions.columns)


    def _invoke_per_tag(self, df: pd.DataFrame, tag_name: str, ts: dict[str, list[object | None]]):
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

        for i in range(len(transforms)):
            transform = transforms[i]
            carry, sub_ts[i] = transform(carry, sub_ts[i])

        return carry.transform_data


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ts = pf.temporal_state.get(StageCode.AC, {})
        assert isinstance(ts, dict)

        tag_names = list(self._components.keys())

        if len(tag_names) == 0:
            return pf

        transformed_parts = [
            self._invoke_per_tag(pf.data, tag_name, ts)
            for tag_name in tag_names
        ]

        # put resultant data on PipeFrame
        pf.actions = pd.concat(transformed_parts, axis=1, copy=False)

        # put new temporal state on PipeFrame
        pf.temporal_state[StageCode.AC] = ts
        return pf


    def reset(self) -> None:
        for transforms in self._components.values():
            for transform in transforms:
                transform.reset()
