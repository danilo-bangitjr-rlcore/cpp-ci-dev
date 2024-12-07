from collections import defaultdict
import pandas as pd
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group, Transform

# ensure transforms are registered
import corerl.data_pipeline.transforms.norm # noqa: F401
import corerl.data_pipeline.transforms.trace # noqa: F401



type SC_TS = dict[
    # tag name
    str,
    # transform steps
    list[object | None],
]

class StateConstructor:
    def __init__(self, cfgs: list[BaseTransformConfig]):
        self._components: list[Transform] = [
            transform_group.dispatch(sub_cfg) for sub_cfg in cfgs
        ]


    def state_dim(self, tag_name: str):
        fake_data = pd.DataFrame({ tag_name: [0., 1.] })

        pf = PipelineFrame(
            data=fake_data,
            caller_code=CallerCode.OFFLINE,
            temporal_state=defaultdict(lambda: None),
        )

        pf = self(pf, tag_name)
        return len(pf.data.columns)


    def __call__(self, pf: PipelineFrame, tag_name: str) -> PipelineFrame:
        tag_data = pf.data.get([tag_name])
        assert tag_data is not None

        carry = TransformCarry(
            obs=pf.data,
            transform_data=tag_data.copy(),
            tag=tag_name,
        )

        ts = pf.temporal_state.get(StageCode.SC, None)
        tag_ts = self._sanitize_temporal_state(ts, tag_name)

        for i in range(len(self._components)):
            transform = self._components[i]
            transform_ts = tag_ts[i]

            carry, transform_ts = transform(carry, transform_ts)
            tag_ts[i] = transform_ts

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_name, axis=1, inplace=False)
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
