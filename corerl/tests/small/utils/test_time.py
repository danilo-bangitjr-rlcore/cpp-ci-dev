from datetime import datetime, timedelta

import corerl.utils.time as time_util


def test_split_into_chunks():
    start = datetime(2024, 8, 1, 1)
    end = datetime(2024, 8, 1, 5, 30)
    Δ = timedelta(hours=1)

    chunks = time_util.split_into_chunks(start, end, Δ)

    assert list(chunks) == [
        (datetime(2024, 8, 1, 1), datetime(2024, 8, 1, 2)),
        (datetime(2024, 8, 1, 2), datetime(2024, 8, 1, 3)),
        (datetime(2024, 8, 1, 3), datetime(2024, 8, 1, 4)),
        (datetime(2024, 8, 1, 4), datetime(2024, 8, 1, 5)),
        (datetime(2024, 8, 1, 5), datetime(2024, 8, 1, 5, 30)),
    ]


def test_exclude_from_chunk_single_no_overlap_after():
    """Test when exclude chunk has no overlap with time chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunk = (datetime(2024, 1, 1, 14), datetime(2024, 1, 1, 16))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert result == [(datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))]


def test_exclude_from_chunk_single_no_overlap_before():
    """Test when exclude chunk is before time chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunk = (datetime(2024, 1, 1, 8), datetime(2024, 1, 1, 9))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert result == [(datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))]


def test_exclude_from_chunk_single_complete_overlap():
    """Test when exclude chunk completely contains time chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunk = (datetime(2024, 1, 1, 9), datetime(2024, 1, 1, 13))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert not result


def test_exclude_from_chunk_single_exact_overlap():
    """Test when exclude chunk exactly matches time chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert not result


def test_exclude_from_chunk_single_split_middle():
    """Test when exclude chunk is inside time chunk, splitting it in two."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 16))
    exclude_chunk = (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 14))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    expected = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 14), datetime(2024, 1, 1, 16)),
    ]
    assert result == expected


def test_exclude_from_chunk_single_partial_overlap_start():
    """Test when exclude chunk overlaps at the start of time chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 14))
    exclude_chunk = (datetime(2024, 1, 1, 8), datetime(2024, 1, 1, 12))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert result == [(datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 14))]


def test_exclude_from_chunk_single_partial_overlap_end():
    """Test when exclude chunk overlaps at the end of time chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 14))
    exclude_chunk = (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 16))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert result == [(datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))]


def test_exclude_from_chunk_single_touching_boundaries():
    """Test when exclude chunk touches time chunk boundaries."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunk = (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 14))

    result = time_util.exclude_from_chunk_single(time_chunk, exclude_chunk)

    assert result == [(datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))]


def test_exclude_from_chunk_empty_excludes():
    """Test exclude_from_chunk with no exclusions."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunks = []

    result = time_util.exclude_from_chunk(time_chunk, exclude_chunks)

    assert result == [(datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))]


def test_exclude_from_chunk_single_exclusion():
    """Test exclude_from_chunk with one exclusion that splits the chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 16))
    exclude_chunks = [(datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 14))]

    result = time_util.exclude_from_chunk(time_chunk, exclude_chunks)

    expected = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 14), datetime(2024, 1, 1, 16)),
    ]
    assert result == expected


def test_exclude_from_chunk_multiple_exclusions():
    """Test exclude_from_chunk with multiple exclusions."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 20))
    exclude_chunks = [
        (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 13)),
        (datetime(2024, 1, 1, 15), datetime(2024, 1, 1, 17)),
    ]

    result = time_util.exclude_from_chunk(time_chunk, exclude_chunks)

    expected = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 13), datetime(2024, 1, 1, 15)),
        (datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 20)),
    ]
    assert result == expected


def test_exclude_from_chunk_overlapping_exclusions():
    """Test exclude_from_chunk with overlapping exclusions."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 20))
    exclude_chunks = [
        (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 15)),
        (datetime(2024, 1, 1, 14), datetime(2024, 1, 1, 17)),
    ]

    result = time_util.exclude_from_chunk(time_chunk, exclude_chunks)

    expected = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 20)),
    ]
    assert result == expected


def test_exclude_from_chunk_complete_exclusion():
    """Test exclude_from_chunk where exclusion completely covers the chunk."""
    time_chunk = (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12))
    exclude_chunks = [(datetime(2024, 1, 1, 9), datetime(2024, 1, 1, 13))]

    result = time_util.exclude_from_chunk(time_chunk, exclude_chunks)

    assert not result


def test_exclude_from_chunks_empty_lists():
    """Test exclude_from_chunks with empty input lists."""
    result = time_util.exclude_from_chunks([], [])
    assert not result


def test_exclude_from_chunks_empty_excludes():
    """Test exclude_from_chunks with no exclusions."""
    time_chunks = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 14), datetime(2024, 1, 1, 16)),
    ]
    exclude_chunks = []

    result = time_util.exclude_from_chunks(time_chunks, exclude_chunks)

    assert result == time_chunks


def test_exclude_from_chunks_single_chunk():
    """Test exclude_from_chunks with single chunk and exclusion."""
    time_chunks = [(datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 16))]
    exclude_chunks = [(datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 14))]

    result = time_util.exclude_from_chunks(time_chunks, exclude_chunks)

    expected = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 14), datetime(2024, 1, 1, 16)),
    ]
    assert result == expected


def test_exclude_from_chunks_multiple_chunks():
    """Test exclude_from_chunks with multiple chunks and exclusions."""
    time_chunks = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 14)),
        (datetime(2024, 1, 1, 16), datetime(2024, 1, 1, 20)),
    ]
    exclude_chunks = [
        (datetime(2024, 1, 1, 11), datetime(2024, 1, 1, 12)),
        (datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 19)),
    ]

    result = time_util.exclude_from_chunks(time_chunks, exclude_chunks)

    expected = [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 11)),
        (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 14)),
        (datetime(2024, 1, 1, 16), datetime(2024, 1, 1, 17)),
        (datetime(2024, 1, 1, 19), datetime(2024, 1, 1, 20)),
    ]
    assert result == expected
