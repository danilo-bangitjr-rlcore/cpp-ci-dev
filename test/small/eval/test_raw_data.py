import numpy as np
import pandas as pd
import pytest

from corerl.eval.raw_data import length_of_chunks, raw_data_eval_for_tag


def test_length_of_chunks_all_true():
    series = pd.Series([True, True, True, True])
    assert length_of_chunks(series) == [4]


def test_length_of_chunks_all_false():
    series = pd.Series([False, False, False, False])
    assert length_of_chunks(series) == [0]


def test_length_of_chunks_mixed():
    series = pd.Series([True, True, False, True, False, True, True, True])
    assert length_of_chunks(series) == [2, 1, 3]


def test_length_of_chunks_single_true():
    series = pd.Series([True])
    assert length_of_chunks(series) == [1]


def test_length_of_chunks_single_false():
    series = pd.Series([False])
    assert length_of_chunks(series) == [0]


def test_length_of_chunks_empty_series():
    series = pd.Series([])
    assert length_of_chunks(series) == [0]


@pytest.fixture
def sample_dataframe():
    data = {
        'tag1': [1, 2, 3, np.nan, 5, np.nan, 7, 8, 9, 10],
        'tag2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'tag3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    return pd.DataFrame(data)


def test_raw_data_eval_for_tag(sample_dataframe: pd.DataFrame):
    """
    Tests to see if the raw data eval is returning and of the right type. Does not check for correctness.
    """
    for tag in sample_dataframe.columns:
        return_dict = raw_data_eval_for_tag(sample_dataframe, tag)
        for k, v in return_dict.items():
            assert isinstance(k, str) and (isinstance(v, float) or isinstance(v, int))

