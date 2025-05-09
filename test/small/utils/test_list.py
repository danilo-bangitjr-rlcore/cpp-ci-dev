import pytest

import corerl.utils.list as list_u


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
        ]
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

