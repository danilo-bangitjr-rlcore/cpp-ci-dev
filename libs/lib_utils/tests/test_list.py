import pytest

import lib_utils.list as list_u


# ----------
# -- find --
# ----------
def test_find_returns_first_match():
    """
    Test that find returns the first matching element.
    """
    items = [1, 2, 3, 4, 5]
    result = list_u.find(lambda x: x > 2, items)
    assert result == 3


def test_find_returns_none_when_no_match():
    """
    Test that find returns None when no element matches.
    """
    items = [1, 2, 3]
    result = list_u.find(lambda x: x > 10, items)
    assert result is None


# ----------------
# -- find_index --
# ----------------
def test_find_index_returns_first_match_index():
    """
    Test that find_index returns the index of the first matching element.
    """
    items = [1, 2, 3, 4, 3]
    result = list_u.find_index(lambda x: x == 3, items)
    assert result == 2


def test_find_index_returns_none_when_no_match():
    """
    Test that find_index returns None when no element matches.
    """
    items = [1, 2, 3]
    result = list_u.find_index(lambda x: x > 10, items)
    assert result is None


# -------------------
# -- find_instance --
# -------------------
def test_find_instance_returns_first_matching_type():
    """
    Test that find_instance returns the first instance of the requested type.
    """
    items = [1, "hello", 2.5, "world"]
    result = list_u.find_instance(str, items)
    assert result == "hello"


def test_find_instance_returns_none_when_no_match():
    """
    Test that find_instance returns None when the type is not found.
    """
    items = [1, 2, 3]
    result = list_u.find_instance(str, items)
    assert result is None


def test_find_instance_with_subclass():
    """
    Test that find_instance works with subclass instances.
    """

    class Animal:
        pass

    class Dog(Animal):
        pass

    items = [1, Dog(), "text"]
    result = list_u.find_instance(Animal, items)
    assert isinstance(result, Dog)


# ---------------------
# -- filter_instance --
# ---------------------
def test_filter_instance_returns_all_matching_types():
    """
    Test that filter_instance returns all instances of the requested type.
    """
    items = [1, "hello", 2, "world", 3.5]
    result = list_u.filter_instance(str, items)
    assert result == ["hello", "world"]


def test_filter_instance_returns_empty_list_when_no_match():
    """
    Test that filter_instance returns an empty list when the type is not found.
    """
    items = [1, 2, 3]
    result = list_u.filter_instance(str, items)
    assert result == []


def test_filter_instance_with_subclasses():
    """
    Test that filter_instance works with subclass instances.
    """

    class Animal:
        pass

    class Dog(Animal):
        pass

    class Cat(Animal):
        pass

    dog1, dog2 = Dog(), Dog()
    cat = Cat()
    items = [1, dog1, "text", cat, dog2]
    result = list_u.filter_instance(Animal, items)
    assert len(result) == 3
    assert dog1 in result
    assert dog2 in result
    assert cat in result


# -------------
# -- flatten --
# -------------
def test_flatten1():
    li = [1, [2], [[3]]]

    got = list_u.flatten(li)
    expected = [1, 2, 3]

    assert got == expected


def test_flatten2():
    li = [1, 2, 3]

    got = list_u.flatten(li)
    expected = [1, 2, 3]

    assert got == expected


# ---------------
# -- partition --
# ---------------
def test_partition():
    left, right = list_u.partition(
        lambda a: a < 3,
        [0, 1, 2, 3, 4, 5, 6],
    )

    assert left == [0, 1, 2]
    assert right == [3, 4, 5, 6]


# ----------------------
# -- multi_level_sort --
# ----------------------
def test_multi_level_sort():
    items = [
        'a.1',
        'b.1',
        'a.6',
        'b.4',
        'a.3',
        'b.2',
    ]

    got = list_u.multi_level_sort(
        items,
        categories=[
            # level 1: letters
            lambda t: t.split('.')[0] == 'a',
            # level 2: numbers
        ],
    )

    assert got == [
        'a.1',
        'a.3',
        'a.6',
        'b.1',
        'b.2',
        'b.4',
    ]



# ----------------------
# --    sort by       --
# ----------------------

def test_sort_by():
    l1 = ['b', 'a', 'c']
    l2 = [1, 0, 2]
    got1, got2 = list_u.sort_by(l1, l2)

    assert got1 == ['a', 'b', 'c']
    assert got2 == [0, 1, 2]


def test_sort_by_fail():
    l1 = ['b', 'a', 'c']
    l2 = [1, 0, 2, 3] # too many elements, so sort_by should fail

    with pytest.raises(ValueError):
        _ = list_u.sort_by(l1, l2)
