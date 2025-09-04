import pandas as pd

from corerl.utils.pandas import split_dataframe_into_chunks


def test_split_dataframe_into_chunks():
    """Test the DataFrame chunking utility function"""
    # Create test dataframe
    df = pd.DataFrame({'col1': range(25), 'col2': range(25, 50)})

    # Test chunking
    chunks = split_dataframe_into_chunks(df, 10)

    assert len(chunks) == 3  # 25 rows / 10 = 3 chunks (last chunk has 5 rows)
    assert len(chunks[0]) == 10
    assert len(chunks[1]) == 10
    assert len(chunks[2]) == 5

    # Verify data integrity
    assert chunks[0]['col1'].tolist() == list(range(10))
    assert chunks[1]['col1'].tolist() == list(range(10, 20))
    assert chunks[2]['col1'].tolist() == list(range(20, 25))


def test_split_dataframe_into_chunks_smaller_than_chunk():
    """Test chunking when dataframe is smaller than chunk size"""
    df = pd.DataFrame({'col1': range(5)})
    chunks = split_dataframe_into_chunks(df, 10)

    assert len(chunks) == 1
    assert len(chunks[0]) == 5
    assert chunks[0]['col1'].tolist() == [0, 1, 2, 3, 4]
