import pytest
import numpy as np

from corerl.data.normalizer_utils import AvgNanNorm, InvertibleNormalizer, Clip, Identity, MaxMin, OneHot, Scale



# ---------------------
# -- All Normalizers --
# ---------------------

INVERTIBLE_NORMALIZERS: list[InvertibleNormalizer] = [
    Identity(),
    AvgNanNorm(min=-1, max=1),
    MaxMin(1.1, 4.4),
    # OneHot(total_count=4, start_from=-22),
    Scale(2.2, 3.3),
]

@pytest.mark.parametrize('norm', INVERTIBLE_NORMALIZERS)
def test_invertible1(norm: InvertibleNormalizer):
    """
    All normalizers should be invertible by
    calling `denormalize` after normalizing.
    """

    x = np.random.uniform(-22, 22, size=(10, 4))
    assert np.allclose(norm.denormalize(norm(x)), x)


# --------------
# -- Identity --
# --------------

def test_identity1():
    """
    The identity normalizer does not change
    floating point input values.
    """

    norm = Identity()
    x = 2.2

    assert norm(x) == x
    assert norm.denormalize(x) == x


def test_identity2():
    """
    The identity normalizer does not change
    numpy input values.
    """

    norm = Identity()
    x = np.array([2.2, 3.3, 4.4])

    assert np.allclose(norm(x), x)
    assert np.allclose(norm.denormalize(x), x)


# -----------
# -- Scale --
# -----------

def test_scale1():
    """
    The scale normalizer centers input floating
    point values.
    """

    norm = Scale(scale=1.0, bias=2.2)

    x = 4.4
    assert norm(x) == 2.2
    assert norm.denormalize(2.2) == x


def test_scale2():
    """
    The scale normalizer centers input numpy
    array values.
    """

    norm = Scale(scale=1.0, bias=2.2)

    x = np.array([2.2, 4.2, 6.2])

    expect = np.array([0., 2., 4.])
    assert np.allclose(norm(x), expect)
    assert np.allclose(norm.denormalize(expect), x)

def test_scale3():
    """
    The scale normalizer centers and scales
    input numpy array values.
    """

    norm = Scale(scale=2.0, bias=2.2)

    x = np.array([2.2, 4.2, 6.2])

    expect = np.array([0., 1., 2.])
    assert np.allclose(norm(x), expect)
    assert np.allclose(norm.denormalize(expect), x)


# ----------
# -- Clip --
# ----------

def test_clip1():
    """
    The clip normalizer restricts input values to a range.
    """

    norm = Clip(-1, 1)

    x = np.array([-2, -1, 0, 1, 2])
    expect = np.array([-1, -1, 0, 1, 1])

    assert np.allclose(norm(x), expect)


# -------------
# -- One Hot --
# -------------

def test_onehot1():
    """
    The onehot normalizer takes a batch of size N
    with a single "feature" dimension of size 1
    and creates a one-hot representation of size
    (N, num_possible_outcomes).
    """

    norm = OneHot(
        total_count=10,
        start_from=0,
    )

    x = np.array([[0, 1, 2, 3]]).T
    expect = np.zeros((4, 10))
    expect[0, 0] = 1
    expect[1, 1] = 1
    expect[2, 2] = 1
    expect[3, 3] = 1

    got = norm(x)
    assert np.allclose(got, expect)
    assert np.allclose(norm.denormalize(got), x)


# ----------------
# -- AvgNanNorm --
# ----------------

def test_avgnannorm1():
    """
    """

    norm = AvgNanNorm(min=-5, max=5)

    x = np.array([
        [np.nan,      0,     1],
        [np.nan, np.nan,     2],
        [np.nan,      2,     3],
        [np.nan, np.nan, np.nan],
    ])
    got = norm(x)

    expect = np.array([
        [0.5, 0.5, 0.6],
        [0.5, 0.5, 0.7],
        [0.5, 0.7, 0.8],
        [0.5, 0.7, 0.8],
    ])

    assert np.allclose(got, expect)
    assert np.allclose(norm.denormalize(got), x)
