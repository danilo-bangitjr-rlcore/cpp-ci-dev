from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from types import EllipsisType
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import register_pytree_node_class
from numpy.typing import NDArray

from lib_utils.jax_timestamp import JaxTimestamp
from lib_utils.maybe import Maybe


def maybe_expand_dim0[A: (NDArray, jax.Array)](arr: A) -> A:
    if arr.ndim == 1:
        if isinstance(arr, jax.Array):
            arr = jnp.expand_dims(arr, 0)
        else:
            arr = np.expand_dims(arr, 0)
    return arr

@register_pytree_node_class
class NamedArray:  # noqa: PLW1641
    def __init__(
        self,
        names: Sequence[str],
        values: jax.Array | np.ndarray,
        timestamps: NDArray[np.datetime64] | None = None,
    ):
        # if no timestamps provided, create timezone naive datetime with value from UTC
        ts = Maybe(timestamps).or_else(
            np.full(values.shape[:-1], np.datetime64(datetime.now(UTC).replace(tzinfo=None))),
        )
        values = jnp.asarray(values)
        self._input_validation(names, values, ts)
        self._names = tuple(names)
        self._values = values
        self._timestamps = JaxTimestamp.from_datetime64(ts)
        self._known_names: set[str] = set(self._names)

    def _input_validation(self, names: Sequence[str], values: jax.Array, timestamps: NDArray[np.datetime64]):
        assert len(names) == len(set(names)), "names must be unique"
        assert len(names) == values.shape[-1], "len of provided names must match number of features"
        assert timestamps.shape == values.shape[:-1], "provided timestamps must match shape of values"


    # properties
    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def array(self) -> jax.Array:
        return self._values

    @property
    def names(self) -> Sequence[str]:
        return self._names

    @property
    def timestamps(self):
        return self._timestamps


    # constructors
    @classmethod
    def from_mapping(cls, data: Mapping[str, jax.Array], timestamps: NDArray[np.datetime64] | None = None) -> Self:
        names = tuple([str(k) for k in data.keys()])
        vals = jnp.concatenate([jnp.expand_dims(data[name], -1) for name in names], axis=-1)
        return cls(names, vals, timestamps)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Self:

        timestamps = (
            Maybe(df.index)
            .is_instance(pd.DatetimeIndex)
            .map(lambda idx: idx.tz_convert(None) if idx.tz is not None else idx)
            .map(lambda idx: np.asarray(idx, dtype='datetime64[us]'))
        )
        arr = jnp.asarray(df)

        return cls(df.columns.tolist(), arr, timestamps.unwrap())

    @classmethod
    def unnamed(cls, vals: jax.Array | np.ndarray) -> Self:
        dummy_names = [str(i) for i in range(vals.shape[-1])]
        return cls(dummy_names, vals)


    # translation
    def as_mapping(self) -> Mapping[str, jax.Array]:
        return {k: self.get_feature(k).expect() for k in self.names}

    def as_pandas(self) -> Maybe[pd.DataFrame]:
        # TODO: use timestamps
        if self.ndim > 2:
            return Maybe(None)

        timestamps = self.timestamps.to_datetime64()
        timestamps = np.expand_dims(timestamps, 0) if timestamps.ndim == 0 else timestamps
        df = pd.DataFrame(
            data=maybe_expand_dim0(self.array),
            columns=pd.Index(self.names),
            index=pd.DatetimeIndex(timestamps),
        )
        return Maybe(df)

    def __array__(self) -> np.ndarray:
        return np.asarray(self._values)

    def __jax_array__(self) -> jax.Array:
        return self._values


    # accessors
    def get_feature(self, key: str, /) -> Maybe[jax.Array]:
        """
        NamedArray uses the convention that the last dimension of a tensor is the "feature" dimension.
        E.g. semantically labeled dims would look like (ensemble, batch, feature)
        """
        if key not in self._known_names:
            return Maybe[jax.Array](None)
        idx = self._names.index(key)
        return Maybe(self._values[..., idx])

    def get_features(self, keys: Iterable[str]) -> Maybe[jax.Array]:
        if not set(keys).issubset(self._known_names):
            return Maybe[jax.Array](None)
        return Maybe(jnp.concat([jnp.expand_dims(self.get_feature(k).expect(), -1) for k in keys], axis=-1))

    def __getitem__(self, key: int | slice | tuple[int | slice | EllipsisType, ...]):
        if self.array.ndim == 1:
            raise IndexError("NamedArray must have more than one dim to be indexed")

        new_narr = NamedArray(self.names, self.array[key])
        # workaround to avoid working with numpy datetime64 in jax transforms
        new_high_bits = maybe_expand_dim0(self._timestamps.high_bits[key])
        new_low_bits = maybe_expand_dim0(self._timestamps.low_bits[key])
        new_narr._timestamps = JaxTimestamp(new_high_bits, new_low_bits)
        return new_narr


    # arithmetic
    def set(self, values: jax.Array) -> Self:
        """
        method to update the values of the stored array.
        Useful to create a new NamedArray with the same feature names and timestamps, but new values
        """
        return self.__class__(self.names, values, self.timestamps.to_datetime64())

    def add(self, other: Self) -> Self:
        """
        not commutative: keeps timestamps of self
        """
        assert other.names == self.names
        n_features = self.array.shape[-1]
        assert other.array.shape[-1] == n_features, "number of features must match"
        assert other.array.shape in {self.array.shape, (n_features,), (1, n_features)}
        return self.set(self.array + other.array)

    def __add__(self, other: float) -> Self:
        """
        Supports scalar addition. To add with another NamedArray, use self.add()
        """
        return self.set(self.array + other)

    def sub(self, other: Self) -> Self:
        """
        not commutative: keeps timestamps of self
        """
        assert other.names == self.names
        n_features = self.array.shape[-1]
        assert other.array.shape[-1] == n_features, "number of features must match"
        assert other.array.shape in {self.array.shape, (n_features,), (1, n_features)}
        return self.set(self.array - other.array)

    def __sub__(self, other: float) -> Self:
        """
        Supports scalar subtraction. To subtract with another NamedArray, use self.sub()
        """
        return self.set(self.array - other)

    def mul(self, other: Self) -> Self:
        """
        not commutative: keeps timestamps of self
        """
        assert other.names == self.names
        n_features = self.array.shape[-1]
        assert other.array.shape[-1] == n_features, "number of features must match"
        assert other.array.shape in {self.array.shape, (n_features,), (1, n_features)}
        return self.set(self.array * other.array)

    def __mul__(self, other: float) -> Self:
        """
        Supports scalar multiplication. To multiply with another NamedArray, use self.mul()
        """
        return self.set(self.array * other)

    def div(self, other: Self) -> Self:
        """
        not commutative: keeps timestamps of self
        """
        assert other.names == self.names
        n_features = self.array.shape[-1]
        assert other.array.shape[-1] == n_features, "number of features must match"
        assert other.array.shape in {self.array.shape, (n_features,), (1, n_features)}
        return self.set(self.array / other.array)

    def __truediv__(self, other: float) -> Self:
        """
        Supports scalar division. To divide with another NamedArray, use self.div()
        """
        return self.set(self.array / other)

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, NamedArray):
            return False
        if (
            value.names != self.names
            or (value.array != self.array).all()
            or value.timestamps != self.timestamps
        ):
            return False
        return True


    # methods for compatibility with jax transforms
    def tree_flatten(self):
        children = (self.array, self._timestamps)
        aux_data = (self._names)
        return (children , aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any):
        # avoid calling __init__ within jax transform internals: see https://docs.jax.dev/en/latest/pytrees.html#custom-pytrees-and-initialization
        obj = object.__new__(NamedArray)
        obj._names = aux_data
        obj._values = children[0]
        obj._timestamps = children[1]
        obj._known_names = set(aux_data)
        return obj


    # Collection methods
    def __contains__(self, key: object, /) -> bool:
        return key in self._names

    def __len__(self) -> int:
        return len(self._values)


    # Pretty print
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        vals = self.array
        if vals.ndim > 2:
            return 'WARNING: Attempted to print NamedArray with ndim > 2'

        n = len(self._names)
        w = min(max(len(name) for name in self._names) + 2, 10)
        total_width = n * (w + 1) + 1
        v = '\u2502' # vertical line
        h = '\u2500' * total_width + '\n' # horizontal line
        mid_h = v + '\u2500' * (total_width - 2) + v + '\n' # horizontal line with vertical edges (wide H shape)
        frame = h + v + v.join([name.center(w) for name in self._names]) + v + '\n' + mid_h

        def get_val_line(row: jax.Array):
            return v + v.join([f"{val:.2f}".center(w) for val in row]) + v + "\n"

        if vals.ndim == 1:
            return frame + get_val_line(vals) + h

        for row in vals:
            frame = frame + get_val_line(row) + mid_h

        return frame[:-(total_width+1)] + h # replace last midline
