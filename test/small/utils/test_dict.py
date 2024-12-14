import hashlib
from typing import Any
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


# -----------
# -- merge --
# -----------

def test_merge1():
    d1 = {
        'a': 1,
        'b': 2,
        'd': 4,
    }

    d2 = {
        'a': 5,
        'c': 3,
    }

    # right-most dict takes precedence
    got = dict_u.merge(d1, d2)
    assert got == {
        'a': 5,
        'b': 2,
        'c': 3,
        'd': 4,
    }

    # no dicts are harmed in the merging
    assert d1 == {
        'a': 1,
        'b': 2,
        'd': 4,
    }
    assert d2 == {
        'a': 5,
        'c': 3,
    }


def test_merge2():
    d1 = {
        'a': 1,
        'b': {
            'c': 2,
            'd': ['hi', 'there'],
        },
        'f': [1, 2, 3],
    }

    d2 = {
        'a': 5,
        'b': {
            'c': 3,
            'e': 4,
        },
        'f': [4, 5, 6],
    }

    # right-most dict takes precedence
    got = dict_u.merge(d1, d2)
    assert got == {
        'a': 5,
        'b': {
            'c': 3,
            'd': ['hi', 'there'],
            'e': 4,
        },
        'f': [4, 5, 6],
    }


def test_merge3():
    d1 = {
        'list': [
            {'a': 22},
            {'b': 33},
        ],
    }

    d2 = {
        'list': [
            {'a': 44},
            {'c': 55},
        ],
    }

    got = dict_u.merge(d1, d2)
    assert got == {
        'list': [
            {'a': 44},
            {'b': 33, 'c': 55},
        ],
    }


# -----------------
# -- set_at_path --
# -----------------

def test_set_at_path1():
    d = {}
    path = 'a.b.c'
    got = dict_u.set_at_path(d, path, val=22)

    assert got == {
        'a': {
            'b': {
                'c': 22,
            },
        },
    }

def test_set_at_path2():
    d: dict[str, Any] = { 'a': {} }
    path = 'a.b.c'
    got = dict_u.set_at_path(d, path, val=22)

    assert got == {
        'a': {
            'b': {
                'c': 22,
            },
        },
    }

def test_set_at_path3():
    d: dict[str, Any] = {}
    path = 'a.b[1].c'
    got = dict_u.set_at_path(d, path, val=22)

    assert got == {
        'a': {
            'b': [
                {},
                {'c': 22},
            ],
        },
    }

def test_set_at_path4():
    d: dict[str, Any] = {
        'ls': [
            { 'name': 'thing', 'val': 1 },
        ]
    }
    path = 'ls[0].name'
    got = dict_u.set_at_path(d, path, val='hello')

    assert got == {
        'ls': [
            { 'name': 'hello', 'val': 1 },
        ],
    }
