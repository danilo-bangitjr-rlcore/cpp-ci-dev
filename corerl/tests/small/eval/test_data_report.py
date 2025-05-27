from math import isclose

import numpy as np
import pandas as pd

from corerl.eval.data_report import correlate, cross_correlation, standardize

# --------------------------- standardization tests -------------------------- #

def test_standardize_all_masked():
    x = np.array([1, 2, 3, 4, 5])
    mask = np.array([True, True, True, True, True])
    result = standardize(x, mask)
    expected = (x - np.mean(x)) / np.std(x)
    print(result)
    print(expected)
    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_none_masked():
    x = np.array([1, 2, 3, 4, 5])
    mask = np.array([False, False, False, False, False])
    result = standardize(x, mask)
    expected = np.zeros_like(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_some_masked():
    x = np.array([1, 0, 1, 0, 4])
    mask = np.array([True, False, True, False, True])
    result = standardize(x, mask)
    expected = np.array([-0.7071067811,
                         0,
                         -0.7071067811,
                         0,
                         1.41421356237])
    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_single_value():
    x = np.array([5])
    mask = np.array([True])
    result = standardize(x, mask)
    expected = np.array([0])
    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_all_ones():
    x = np.array([1, 1, 1, 1, 1])
    mask = np.array([True, True, True, True, True])
    result = standardize(x, mask)
    expected = np.zeros_like(x)
    np.testing.assert_array_almost_equal(result, expected)


# ------------------------------ correlate tests ----------------------------- #

def test_correlate_no_lag():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    max_lag = 0
    result = correlate(x, y, max_lag)
    expected = np.array([55]) # dot product of x with itself
    np.testing.assert_array_almost_equal(result, expected)


def test_correlate_lag():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    max_lag = 2
    result = correlate(x, y, max_lag)
    expected = np.array([26, 40, 55, 40, 26])
    np.testing.assert_array_almost_equal(result, expected)


def test_correlate_counting():
    x = np.array([1, 0, 1, 0, 1])
    y = np.array([1, 1, 1, 0, 1])
    max_lag = 2
    result = correlate(x, y, max_lag)
    expected = np.array([2, 1, 3, 1, 2])
    np.testing.assert_array_almost_equal(result, expected)


# ----------------------------- cross_correlation ---------------------------- #

def test_cross_correlation_no_lag():
    df = pd.DataFrame({
        'tag_1': [1, 2, 3, 4, 5],
        'tag_2': [1, 2, 3, 4, 5]
    })
    max_lag = 2
    max_cc, lag, cc  = cross_correlation(df, 'tag_1', 'tag_2', max_lag)
    assert isclose(max_cc, 1.)
    assert lag == 0
    expected_cc = np.array([
        -0.166666,
        0.5,
        1.0,
        0.5,
        -0.166666,
    ])
    np.testing.assert_array_almost_equal(cc,  expected_cc)


def test_cross_correlation_all_nan():
    df = pd.DataFrame({
        'tag_1': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'tag_2': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    max_lag = 2
    max_cc, lag, cc = cross_correlation(df, 'tag_1', 'tag_2', max_lag)
    assert max_cc == -np.inf
    assert lag == 0
    expected_cc = np.array([-np.inf])
    np.testing.assert_array_almost_equal(cc,  expected_cc)


def test_cross_correlation_partial_nan():
    df = pd.DataFrame({
        'tag_1': [1, 2, np.nan, 4, 5], # standardized: np.array([-1.26491106, -0.63245553, 0, 0.63245553,  1.26491106])
        'tag_2': [1, np.nan, 3, 4, 5]  # standardized: np.array([-1.52127766, 0, -0.16903085,  0.50709255,  1.18321596])

    })
    max_lag = 1
    max_cc, lag, cc = cross_correlation(df, 'tag_1', 'tag_2', max_lag)
    assert isclose(max_cc, 1.24721913)
    assert lag == 0
    expected_cc = np.array([0.42761799, 1.24721913, 0.4988876])
    np.testing.assert_array_almost_equal(cc,  expected_cc)
