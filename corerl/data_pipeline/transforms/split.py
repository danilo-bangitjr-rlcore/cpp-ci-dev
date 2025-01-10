import copy
import pandas as pd
from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms import SplitConfig


@dataclass
class SplitTemporalState:
    left_state: object | None = None
    right_state: object | None = None


class SplitTransform:
    def __init__(self, cfg: SplitConfig):
        self._cfg = cfg

        self._left = [transform_group.dispatch(xform) for xform in cfg.left]
        self._right = [transform_group.dispatch(xform) for xform in cfg.right]

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, SplitTemporalState | None)

        original_state = carry.transform_data.copy()

        # shallow copy all attributes, but ensure agent_state is a deep copy
        r_carry = copy.copy(carry)
        r_carry.transform_data = carry.transform_data.copy()
        r_state = None if ts is None else ts.right_state
        for xform in self._right:
            r_carry, r_state = xform(r_carry, r_state)

        l_carry = carry
        l_state = None if ts is None else ts.left_state
        for xform in self._left:
            l_carry, l_state = xform(l_carry, l_state)

        # reconcile the two carry objects by concatenating agent
        # state and relying on the fact that all other attributes
        # should be "read-only"
        carry.transform_data = pd.concat((l_carry.transform_data, r_carry.transform_data), axis=1)

        if self._cfg.passthrough:
            dup_cols = set(original_state.columns).intersection(carry.transform_data.columns)
            if dup_cols:
                carry.transform_data.drop(list(dup_cols), axis=1, inplace=True)

            carry.transform_data = pd.concat((carry.transform_data, original_state), axis=1)

        # Note a distinction in behavior between passthrough == False
        # and passthrough == None.
        # If the user did not specify a passthrough preference, then
        # defer to the left/right handlers to decide what is passed.
        # If the user specified "do not passthrough", then filter
        # out the original columns.
        elif self._cfg.passthrough is False:
            orig_cols = set(carry.transform_data.columns).intersection(original_state.columns)
            if orig_cols:
                carry.transform_data.drop(list(orig_cols), axis=1, inplace=True)

        return carry, SplitTemporalState(
            left_state=l_state,
            right_state=r_state,
        )

    def reset(self) -> None:
        pass

transform_group.dispatcher(SplitTransform)
