import copy
import pandas as pd
from dataclasses import dataclass
from omegaconf import MISSING

from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig, sc_group
from corerl.data_pipeline.state_constructors.interface import TransformCarry


@dataclass
class SplitConfig(BaseTransformConfig):
    name: str = 'split'

    left: BaseTransformConfig = MISSING
    right: BaseTransformConfig = MISSING


@dataclass
class SplitTemporalState:
    left_state: object | None = None
    right_state: object | None = None


class SplitTransform:
    def __init__(self, cfg: SplitConfig):
        self._cfg = cfg

        self._left = sc_group.dispatch(cfg.left)
        self._right = sc_group.dispatch(cfg.right)

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, SplitTemporalState | None)

        # shallow copy all attributes, but ensure agent_state is a deep copy
        r_carry = copy.copy(carry)
        r_carry.agent_state = carry.agent_state.copy()

        l_state = None if ts is None else ts.left_state
        l_carry, l_state = self._left(carry, l_state)

        r_state = None if ts is None else ts.right_state
        r_carry, r_state = self._right(r_carry, r_state)

        # reconcile the two carry objects by concatenating agent
        # state and relying on the fact that all other attributes
        # should "read-only"
        carry.agent_state = pd.concat((l_carry.agent_state, r_carry.agent_state), axis=1)

        return carry, SplitTemporalState(
            left_state=l_state,
            right_state=r_state,
        )

sc_group.dispatcher(SplitTransform)
