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
    passthrough: bool | None = None


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

        original_state = carry.agent_state.copy()

        # shallow copy all attributes, but ensure agent_state is a deep copy
        r_carry = copy.copy(carry)
        r_carry.agent_state = carry.agent_state.copy()

        l_state = None if ts is None else ts.left_state
        l_carry, l_state = self._left(carry, l_state)

        r_state = None if ts is None else ts.right_state
        r_carry, r_state = self._right(r_carry, r_state)

        # reconcile the two carry objects by concatenating agent
        # state and relying on the fact that all other attributes
        # should be "read-only"
        carry.agent_state = pd.concat((l_carry.agent_state, r_carry.agent_state), axis=1)

        if self._cfg.passthrough:
            dup_cols = set(original_state.columns).intersection(carry.agent_state.columns)
            if dup_cols:
                carry.agent_state.drop(list(dup_cols), axis=1, inplace=True)

            carry.agent_state = pd.concat((carry.agent_state, original_state), axis=1)

        # Note a distinction in behavior between passthrough == False
        # and passthrough == None.
        # If the user did not specify a passthrough preference, then
        # defer to the left/right handlers to decide what is passed.
        # If the user specified "do not passthrough", then filter
        # out the original columns.
        elif self._cfg.passthrough is False:
            orig_cols = set(carry.agent_state.columns).intersection(original_state.columns)
            if orig_cols:
                carry.agent_state.drop(list(orig_cols), axis=1, inplace=True)

        return carry, SplitTemporalState(
            left_state=l_state,
            right_state=r_state,
        )

sc_group.dispatcher(SplitTransform)
