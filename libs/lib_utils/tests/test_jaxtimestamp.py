import jax.tree_util as tu
import numpy as np

from lib_utils.jax_timestamp import JaxTimestamp

# A timestamp that is clearly in the past
T_PAST = JaxTimestamp.from_datetime64(np.asarray(np.datetime64('2023-01-01T00:00:00.000000')))
# A more recent timestamp
T_NOW = JaxTimestamp.from_datetime64(np.asarray(np.datetime64('2024-01-01T12:30:00.500000')))
# A timestamp that is the same as T_NOW
T_NOW_SAME = JaxTimestamp.from_datetime64(np.asarray(np.datetime64('2024-01-01T12:30:00.500000')))
# A timestamp that is slightly after T_NOW (different low bits)
T_NOW_PLUS_MICRO = JaxTimestamp.from_datetime64(np.asarray(np.datetime64('2024-01-01T12:30:00.500001')))
# A timestamp that is clearly in the future (different high bits)
T_FUTURE = JaxTimestamp.from_datetime64(np.asarray(np.datetime64('2025-01-01T00:00:00.000000')))


def test_equality():
    """Tests for __eq__ and __ne__."""
    assert T_NOW == T_NOW_SAME
    assert not (T_NOW == T_PAST)
    assert T_NOW != T_PAST
    assert not (T_NOW != T_NOW_SAME)
    # Test comparison with other types
    assert T_NOW != (1, 2)
    assert not (T_NOW == (1, 2))


def test_greater_than():
    """Tests for __gt__."""
    assert T_NOW > T_PAST
    assert T_FUTURE > T_NOW
    assert T_NOW_PLUS_MICRO > T_NOW
    assert not (T_PAST > T_NOW)
    assert not (T_NOW > T_NOW) # noqa: PLR0124
    # Test comparison with other types
    assert not (T_NOW > (1, 2))


def test_less_than():
    """Tests for __lt__."""
    assert T_PAST < T_NOW
    assert T_NOW < T_FUTURE
    assert T_NOW < T_NOW_PLUS_MICRO
    assert not (T_NOW < T_PAST)
    assert not (T_NOW < T_NOW) # noqa: PLR0124
    # Test comparison with other types
    assert not (T_NOW < (1, 2))


def test_greater_than_or_equal():
    """Tests for __ge__."""
    assert T_NOW >= T_PAST
    assert T_FUTURE >= T_NOW
    assert T_NOW >= T_NOW_SAME
    assert T_NOW_PLUS_MICRO >= T_NOW
    assert not (T_PAST >= T_NOW)
    # Test comparison with other types
    assert not (T_NOW >= (1, 2))


def test_less_than_or_equal():
    """Tests for __le__."""
    assert T_PAST <= T_NOW
    assert T_NOW <= T_FUTURE
    assert T_NOW <= T_NOW_SAME
    assert T_NOW <= T_NOW_PLUS_MICRO
    assert not (T_FUTURE <= T_NOW)
    # Test comparison with other types
    assert not (T_NOW <= (1, 2))


def test_tree_flatten_unflatten():
    """Tests that a JaxTimestamp can be flattened and unflattened."""

    # Test with a scalar JaxTimestamp
    leaves, treedef = tu.tree_flatten(T_NOW)
    t_now_unflattened = tu.tree_unflatten(treedef, leaves)
    assert T_NOW == t_now_unflattened

    # Test with a JaxTimestamp containing an 2x3 array of timestamps
    t_array = JaxTimestamp.from_datetime64(
        np.array(
            [
                [
                    '2020-01-01T00:00:00.000000',
                    '2021-01-01T12:30:00.500000',
                    '2022-01-01T00:00:00.000000',
                ],
                [
                    '2023-01-01T00:00:00.000000',
                    '2024-01-01T12:30:00.500000',
                    '2025-01-01T00:00:00.000000',
                ],
            ],
            dtype='datetime64[us]',
        ),
    )
    leaves, treedef = tu.tree_flatten(t_array)
    t_array_unflattened = tu.tree_unflatten(treedef, leaves)
    assert t_array == t_array_unflattened

