import pytest

from lib_utils.maybe import Maybe


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
            lambda y: x + y,
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

def test_find_positive():
    m = Maybe.find(lambda x: x == 2, [1, 2, 3])
    got = m.unwrap()

    assert got == 2

def test_find_negative():
    m = Maybe.find(lambda x: x == 4, [1, 2, 3])
    got = m.unwrap()

    assert got is None


def test_expect_with_string_message():
    m = Maybe[int](None)

    with pytest.raises(Exception) as excinfo:
        m.expect('badness')

    assert 'badness' in str(excinfo.value)


def test_expect_with_exception_instance():
    m = Maybe[int](None)
    class MyErr(Exception):
        pass

    err = MyErr('boom')

    with pytest.raises(MyErr):
        m.expect(err)


def test_flat_from_try_success():
    def inner() -> Maybe[int]:
        return Maybe(33)

    got = Maybe.flat_from_try(inner).unwrap()
    assert got == 33


def test_flat_from_try_inner_none():
    def inner() -> Maybe[int]:
        return Maybe(None)

    got = Maybe.flat_from_try(inner).unwrap()
    assert got is None


def test_flat_from_try_exception():
    class MyErr(Exception):
        pass

    def inner() -> Maybe[int]:
        raise MyErr('boom')
    res = Maybe.flat_from_try(inner)
    assert res.unwrap() is None


def test_find_instance_returns_some_for_matching_type():
    class A:
        pass

    class B:
        pass

    items = [B(), A(), B()]

    maybe_a = Maybe.find_instance(A, items)

    assert maybe_a.is_some()
    assert isinstance(maybe_a.unwrap(), A)


def test_find_instance_returns_none_when_no_match():
    class A:
        pass

    items = [1, 2, 3]

    maybe_a = Maybe.find_instance(A, items)

    assert maybe_a.is_none()
