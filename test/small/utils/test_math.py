import numpy as np

from corerl.utils.math import put_in_range


def test_put_in_range_float():
    got = put_in_range(0.5, old_range=(0, 1), new_range=(-1, 1))
    assert np.isclose(got, 0)

    got = put_in_range(0.5, old_range=(0, 1), new_range=(0, 1))
    assert np.isclose(got, 0.5)

    got = put_in_range(0.5, old_range=(0, 1), new_range=(1, 2))
    assert np.isclose(got, 1.5)


def test_put_in_range_array():
    got = put_in_range(np.array([0.5]), old_range=(0, 1), new_range=(-1, 1))
    assert np.allclose(got, np.array([0]))

    got = put_in_range(np.array([0.5]), old_range=(0, 1), new_range=(0, 1))
    assert np.allclose(got, np.array([0.5]))

    got = put_in_range(np.array([0.5]), old_range=(0, 1), new_range=(1, 2))
    assert np.allclose(got, np.array([1.5]))
