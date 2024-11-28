import pandas as pd
from collections import defaultdict
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.state_constructors.interface import TransformCarry
from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig, sc_group, StateTransform



type SC_TS = dict[
    # tag name
    str,
    # transform steps
    list[object | None],
]

class StateConstructor:
    def __init__(self, cfgs: list[BaseTransformConfig]):
        self._components: list[StateTransform] = [
            sc_group.dispatch(sub_cfg) for sub_cfg in cfgs
        ]


    def __call__(self, pf: PipelineFrame, tag_name: str) -> PipelineFrame:
        tag = pf.data.get([tag_name])
        assert tag is not None

        carry = TransformCarry(
            obs=pf.data,
            agent_state=tag.copy(),
        )

        ts = pf.temporal_state.get(StageCode.SC, None)
        ts = self._sanitize_temporal_state(ts)
        tag_ts = ts[tag_name]

        for i in range(len(self._components)):
            transform = self._components[i]
            transform_ts = tag_ts[i]

            carry, transform_ts = transform(carry, transform_ts)
            tag_ts[i] = transform_ts

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_name, axis=1, inplace=False)
        pf.data = pd.concat((df, carry.agent_state), axis=1, copy=False)

        # put new temporal state on PipeFrame
        pf.temporal_state[StageCode.SC] = ts
        return pf


    def _sanitize_temporal_state(self, ts: object | None) -> SC_TS:
        if ts is None:
            ts = defaultdict(lambda: [None] * len(self._components))

        assert isinstance(ts, dict)
        return ts
