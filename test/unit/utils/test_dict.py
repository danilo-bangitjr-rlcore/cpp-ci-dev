import corerl.utils.dict as dict_u


# ----------
# -- drop --
# ----------

def test_drop1():
    """
    Can remove a single key from a dictionary
    without mutating the original dictionary.
    """
    d = {
        'a': 1,
        'b': 2,
    }

    got = dict_u.drop(d, ['a'])
    assert got == {
        'b': 2,
    }
    assert d == {
        'a': 1,
        'b': 2,
    }


def test_drop2():
    """
    Can remove multiple keys from a dictionary
    without mutating the original dictionary.
    """
    d = {
        'a': 1,
        'b': 2,
        'c': 3,
    }

    got = dict_u.drop(d, ['a', 'c'])
    assert got == {
        'b': 2,
    }
    assert d == {
        'a': 1,
        'b': 2,
        'c': 3,
    }


# -------------
# -- flatten --
# -------------

def test_flatten1():
    """
    Can flatten a nested dict to a single
    dict with merged keys.
    """
    d = {
        'a': {
            'b': 2,
        },
    }

    got = dict_u.flatten(d)
    assert got == {
        'a_b': 2,
    }


def test_flatten2():
    """
    Can flatten a deeply nested dict to a single
    dict with merged keys.
    """
    d = {
        'a': {
            'b': {
                'c': 'hello there',
                'd': 'general kenobi',
            },
        },
    }

    got = dict_u.flatten(d)
    assert got == {
        'a_b_c': 'hello there',
        'a_b_d': 'general kenobi',
    }


def test_flatten3():
    """
    Can flatten a dict with multiple levels of nesting
    to a single dict with merged keys.
    """
    d = {
        'a': {
            'b': {
                'c': 1,
            },
            'd': 2,
        },
        'e': 3,
    }

    got = dict_u.flatten(d)
    assert got == {
        'a_b_c': 1,
        'a_d': 2,
        'e': 3,
    }
