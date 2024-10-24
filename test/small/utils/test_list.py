import corerl.utils.list as list_u

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
