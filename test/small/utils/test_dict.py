import hashlib
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

# ----------
# -- hash --
# ----------

def test_hash1():
    """
    Hash function produces a unique string hash for a
    given dictionary. Two different dictionaries produce
    different hashes, the same dict produces the same.
    """
    d1 = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
    }

    h1 = dict_u.hash(d1)
    assert isinstance(h1, str)

    d2 = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 4,
        },
    }

    h2 = dict_u.hash(d2)
    assert h1 != h2
    assert h1 == dict_u.hash(d1)


def test_hash2():
    """
    Dictionaries can have different structures and
    produce different hashes.
    """
    d1 = {
        'a': 1,
    }

    d2 = {
        'a': 1,
        'b': 2,
    }

    assert dict_u.hash(d1) != dict_u.hash(d2)


def test_hash_ignore1():
    """
    Keys can be ignored from a dictionary, allowing
    two dictionaries with different structures to have
    the same hash, if the differing keys are ignored.
    """
    d1 = {
        'a': 1,
    }

    d2 = {
        'a': 1,
        'b': 2,
    }

    assert dict_u.hash(d1) != dict_u.hash(d2)

    h1 = dict_u.hash(d1)
    h2 = dict_u.hash(d2, ignore={'b'})

    assert h1 == h2


def test_hash_ignore2():
    """
    Deeply nested keys can also be ignored by
    passing an ignore path.
    """

    d1 = {
        'a': {
            'b': 1,
            'c': 2,
        },
    }

    d2 = {
        'a': {
            'b': 1,
            'c': 3,
        },
    }

    assert dict_u.hash(d1) != dict_u.hash(d2)

    h1 = dict_u.hash(d1, ignore={'a.c'})
    h2 = dict_u.hash(d2, ignore={'a.c'})

    assert h1 == h2


def test_hash_insertion_order1():
    """
    The hash function is agnostic to insertion order
    on the underlying dictionaries. By contrast,
    hashing the string representation of the dicts is
    not robust to insertion order.
    """

    d1 = {
        'a': 1,
        'b': 2,
    }

    d2 = {
        'b': 2,
        'a': 1,
    }

    assert d1 == d2

    # hash the string representation of the dicts
    h1 = hashlib.sha1(str(d1).encode('utf-8')).hexdigest()
    h2 = hashlib.sha1(str(d2).encode('utf-8')).hexdigest()
    assert h1 != h2


    h1 = dict_u.hash(d1)
    h2 = dict_u.hash(d2)
    assert h1 == h2


# ---------------
# -- hash_many --
# ---------------

def test_hash_many1():
    """
    Can hash a list of dictionaries.
    Two lists with different dictionaries should
    produce different hashes.
    """
    ds1 = [
        {'a': 1},
        {'b': 2},
    ]

    ds2 = [
        {'a': 1},
        {'b': 3},
    ]

    assert dict_u.hash_many(ds1) == dict_u.hash_many(ds1)
    assert dict_u.hash_many(ds1) != dict_u.hash_many(ds2)
