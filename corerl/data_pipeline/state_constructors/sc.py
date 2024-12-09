from collections import defaultdict
import pandas as pd
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group, Transform

# ensure transforms are registered
import corerl.data_pipeline.transforms.norm # noqa: F401
import corerl.data_pipeline.transforms.trace # noqa: F401
from corerl.data_pipeline.tag_config import TagConfig # noqa: F401



type SC_TS = dict[
    # tag name
    str,
    # transform steps
    list[object | None],
]

class StateConstructor:
    def __init__(self, cfgs: list[TagConfig], default_cfg: list[BaseTransformConfig]):
        # get sc pipeline configs for each tag
        # if a tag has no specified pipeline config, then use the default
        sc_cfgs = {
            tag.name: tag.state_constructor if tag.state_constructor is not None else default_cfg
            for tag in cfgs
            if not tag.is_action
        }

        self._components: dict[str, list[Transform]] = {
            tag_name: self._construct_components(parts)
            for tag_name, parts in sc_cfgs.items()
        }


    def _construct_components(self, sub_cfgs: list[BaseTransformConfig]):
        return [
            transform_group.dispatch(sub_cfg) for sub_cfg in sub_cfgs
        ]


    def state_dim(self):
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
        return len(pf.data.columns)


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
        ts = pf.temporal_state.get(StageCode.SC, {})
        assert isinstance(ts, dict)

        tag_names = list(self._components.keys())

        transformed_parts = [
            self._invoke_per_tag(pf.data, tag_name, ts)
            for tag_name in tag_names
        ]

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_names, axis=1, inplace=False)
        pf.data = pd.concat([df] + transformed_parts, axis=1, copy=False)

        # put new temporal state on PipeFrame
        pf.temporal_state[StageCode.SC] = ts
        return pf
