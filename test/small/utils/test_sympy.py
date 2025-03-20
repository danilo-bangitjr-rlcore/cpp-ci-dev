import pytest
import sympy as sy

from corerl.utils.sympy import (
    _get_tag_names,
    _is_balanced_braces,
    _preprocess_expression_string,
    _preprocess_tag_names,
    is_affine,
    is_expression,
)


def test_is_affine():
    res = is_affine(sy.sympify("a"))
    assert res

    res = is_affine(sy.sympify("2*a"))
    assert res

    res = is_affine(sy.sympify("2*a+1"))
    assert res

    res = is_affine(sy.sympify("2*a+5*b+1"))
    assert res

    res = is_affine(sy.sympify("2*a+5*b+a*b"))
    assert not res

    res = is_affine(sy.sympify("sin(a)"))
    assert not res

    res = is_affine(sy.sympify("a^2"))
    assert not res

    res = is_affine(sy.sympify("1*a+2*b+3*c+4*d+5*e+1"))
    assert res

    res = is_affine(sy.sympify("ketchup"))
    assert res

    res = is_affine(sy.sympify("a/b"))
    assert not res

def test_preprocess_expression_string():
    res = _preprocess_expression_string("2*{tag-0} - 100*{tag-1} + 0.3* {tag-2}")
    assert res == "2*tag_0 - 100*tag_1 + 0.3* tag_2"

def test_get_symbol_names():
    res = _get_tag_names("2*{tag-0} - 100*{tag-1} + 0.3* {tag-2}")
    assert res == ["tag-0", "tag-1", "tag-2"]

    proc_res = _preprocess_tag_names(res)
    assert proc_res == ["tag_0", "tag_1", "tag_2"]

    proc_tag = _preprocess_tag_names("action-0")
    assert proc_tag == "action_0"

def test_is_expression():
    res = is_expression("abc")
    assert not res

    with pytest.raises(ValueError):
        res = is_expression("a{bc")

    with pytest.raises(ValueError):
        res = is_expression("ab}c")

    with pytest.raises(ValueError):
        res = is_expression("}}}}}")

    with pytest.raises(ValueError):
        res = is_expression("a}b{c")

    with pytest.raises(ValueError):
        res = is_expression("{{{{{}")

    res = is_expression("a{b}c")
    assert res


def test_is_balanced():
    res = _is_balanced_braces("abc")
    assert res

    res = _is_balanced_braces("a{bc")
    assert not res

    res = _is_balanced_braces("ab}c")
    assert not res

    res = _is_balanced_braces("}}}}}")
    assert not res

    res = _is_balanced_braces("a{b}c")
    assert res

    res = _is_balanced_braces("a}b{c")
    assert not res

    res = _is_balanced_braces("{{{{{}")
    assert not res
