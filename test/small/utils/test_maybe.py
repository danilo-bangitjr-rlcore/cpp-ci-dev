import pytest

from corerl.utils.maybe import Maybe


def test_map_positive():
    m = Maybe('22')

    # map takes the value '22' and passes it to `int`
    got_m = m.map(int)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert isinstance(got, int)
    assert got == 22

def test_map_negative():
    m = Maybe(None)

    # map does not call the underlying function
    # because the maybe contains a None
    got_m = m.map(int)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert got is None

def test_flat_map_positive():
    inner = Maybe(22)

    m = Maybe(44)

    # inner function itself returns a maybe type
    # which means a typical map would return a
    # Maybe[Maybe[int]].
    # Instead, flat_map flattens the inner Maybe
    # type into the parent, providing a
    # Maybe[int]
    got_m = m.flat_map(
        lambda x: inner.map(
            lambda y: x + y
        ),
    )
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert isinstance(got, int)
    assert got == 66

def test_flat_map_negative():
    inner = Maybe(22)

    m = Maybe(None)
    got_m = m.flat_map(lambda _: inner)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert got is None

def test_otherwise_positive():
    m = Maybe(22)

    got_m = m.otherwise(lambda: 44)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert got == 22

def test_otherwise_negative():
    m = Maybe[int](None)

    got_m = m.otherwise(lambda: 44)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert got == 44

def test_flat_otherwise_positive():
    m = Maybe(22)
    inner = Maybe(44)

    got_m = m.flat_otherwise(lambda: inner)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert got == 22

def test_flat_otherwise_negative():
    m = Maybe[int](None)
    inner = Maybe(44)

    got_m = m.flat_otherwise(lambda: inner)
    got = got_m.unwrap()

    assert isinstance(got_m, Maybe)
    assert got == 44

def test_or_else_positive():
    got = (
        Maybe(22)
        .or_else(44)
    )

    assert got == 22

def test_or_else_negative():
    got = (
        Maybe[int](None)
        .or_else(44)
    )

    assert got == 44

def test_or_else_double_negative():
    with pytest.raises(Exception): # noqa: B017
        other: int | None = None
        (
            Maybe[int](None)
            .or_else(other, 'Uh-oh!')
        )
