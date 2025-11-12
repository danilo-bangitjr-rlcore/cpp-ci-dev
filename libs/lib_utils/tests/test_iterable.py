from lib_utils.iterable import group_by, group_by_key, keep_iterable, partition


def test_partition_basic():
    data = [("a", 1), ("b", 2), ("a", 3)]
    expected = {"a": [1, 3], "b": [2]}
    assert partition(data) == expected

def test_partition_empty():
    assert partition([]) == {}

def test_group_by_basic():
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 30}]
    expected = {"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 30]}
    assert group_by(data) == expected

def test_group_by_empty():
    assert group_by([]) == {}

def test_group_by_key_basic():
    data = [{"category": "fruit", "item": "apple"},
            {"category": "fruit", "item": "banana"},
            {"category": "veggie", "item": "carrot"}]
    expected = {"fruit": ["apple", "banana"], "veggie": ["carrot"]}
    assert group_by_key(data, "category", "item") == expected

def test_group_by_key_empty():
    assert group_by_key([], "category", "item") == {}

def test_keep_iterable_basic():
    data = [{"name": "Alice", "age": 30, "city": "NYC"}, {"name": "Bob", "age": 25, "city": "LA"}]
    keys = ["name", "age"]
    expected = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    assert keep_iterable(data, keys) == expected

def test_keep_iterable_empty():
    assert keep_iterable([], ["name", "age"]) == []


