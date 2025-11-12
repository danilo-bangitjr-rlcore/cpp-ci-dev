from typing import NamedTuple, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_T_co = TypeVar("_T_co", covariant=True)

class JaxTimestamp(NamedTuple):
    high_bits: jax.Array
    low_bits: jax.Array

    def __gt__(self, value: tuple[_T_co, ...], /) -> bool:
        if not isinstance(value, JaxTimestamp):
            return False
        is_gt = (
            (self.high_bits > value.high_bits).all()
            | (
                (self.high_bits == value.high_bits)
                & (self.low_bits > value.low_bits).all()
            )
        )
        return jax.lax.cond(
            is_gt,
            lambda: True,
            lambda: False,
        )

    def __lt__(self, value: tuple[_T_co, ...], /) -> bool:
        if not isinstance(value, JaxTimestamp):
            return False
        is_lt = (
            (self.high_bits < value.high_bits).all()
            | (
                (self.high_bits == value.high_bits)
                & (self.low_bits < value.low_bits).all()
            )
        )
        return jax.lax.cond(
            is_lt,
            lambda: True,
            lambda: False,
        )

    def __le__(self, value: tuple[_T_co, ...], /) -> bool:
        if not isinstance(value, JaxTimestamp):
            return False
        is_le = (
            (self.high_bits < value.high_bits).all()
            | (
                (self.high_bits == value.high_bits)
                & ~(self.low_bits > value.low_bits).all()
            )
        )
        return jax.lax.cond(
            is_le,
            lambda: True,
            lambda: False,
        )

    def __ge__(self, value: tuple[_T_co, ...], /) -> bool:
        if not isinstance(value, JaxTimestamp):
            return False
        is_ge = (
            (self.high_bits > value.high_bits).all()
            | (
                (self.high_bits == value.high_bits)
                & ~(self.low_bits < value.low_bits).all()
            )
        )
        return jax.lax.cond(
            is_ge,
            lambda: True,
            lambda: False,
        )

    @classmethod
    def from_datetime64(cls, t: NDArray[np.datetime64]) -> Self:
        datetime_int = t.astype(np.uint64)

        # Get the most significant (upper) 32 bits by shifting right
        high_bits = jnp.asarray(datetime_int >> 32, dtype=jnp.uint32)

        # Get the least significant (lower) 32 bits using a mask
        # 0xFFFFFFFF is hexadecimal for 2**32 - 1
        low_bits = jnp.asarray(datetime_int & 0xFFFFFFFF, dtype=jnp.uint32)

        return cls(high_bits, low_bits)

    def to_datetime64(self) -> NDArray[np.datetime64]:
        # First, cast the high bits back to uint64 so there's room to shift left
        shifted_high_bits = np.asarray(self.high_bits, dtype=np.uint64) << 32

        # Combine the shifted high bits and the low bits with a bitwise OR
        reconstructed_uint64 = shifted_high_bits | np.asarray(self.low_bits, dtype=np.uint64)

        return reconstructed_uint64.astype('datetime64[us]')
