from lib_utils.iterable import partition


def test_partition_basic():
    data = [("a", 1), ("b", 2), ("a", 3)]
    expected = {"a": [1, 3], "b": [2]}
    assert partition(data) == expected

def test_partition_empty():
    assert partition([]) == {}
