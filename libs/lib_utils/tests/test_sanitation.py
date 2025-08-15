import re

from lib_utils.sql_logging.utils import (
    ColumnMapper,
    SanitizedName,
    _clean_names_with_hash_disambiguation,
    _get_short_hash,
    _sanitize_key,
)


class TestSanitizeKey:
    """Test the basic sanitization function"""

    def test_basic_sanitization(self):
        """Test that special characters are replaced with underscores and converted to lowercase"""
        assert _sanitize_key("My-Name!") == "my_name_"
        assert _sanitize_key("Hello$World") == "hello_world"
        assert _sanitize_key("TEST@123") == "test_123"
        assert _sanitize_key("CamelCase") == "camelcase"

    def test_consecutive_special_characters(self):
        """Test that multiple consecutive special characters collapse to single underscores"""
        assert _sanitize_key("test---name") == "test_name"
        assert _sanitize_key("hello!!!world") == "hello_world"
        assert _sanitize_key("a@#$%b") == "a_b"
        assert _sanitize_key("___test___") == "_test_"

    def test_already_clean_names(self):
        """Test that names with only alphanumeric characters and underscores pass through unchanged"""
        assert _sanitize_key("clean_name") == "clean_name"
        assert _sanitize_key("test123") == "test123"
        assert _sanitize_key("already_clean_123") == "already_clean_123"

    def test_empty_and_whitespace_handling(self):
        """Test empty strings, whitespace-only strings, and strings that become empty"""
        assert _sanitize_key("") == ""
        assert _sanitize_key("   ") == "_"
        assert _sanitize_key("@#$") == "_"
        assert _sanitize_key("  hello  ") == "_hello_"


class TestCleanNamesWithHashDisambiguation:
    """Test the collision detection and hash disambiguation"""

    def test_collision_detection_and_resolution(self):
        """Test that duplicate sanitized names get unique hash suffixes"""
        names = ["my-name", "my!name", "my$name"]
        result = _clean_names_with_hash_disambiguation(names)

        # All should be different
        assert len(set(result)) == 3

        # All should contain the base name
        for name in result:
            assert "my_name" in name

        # Should have hash suffixes
        hash_suffixed = [name for name in result if len(name) > len("my_name")]
        assert len(hash_suffixed) == 3

    def test_no_unnecessary_hashing(self):
        """Test that unique names don't get hash suffixes added"""
        names = ["unique1", "unique2", "unique3"]
        result = _clean_names_with_hash_disambiguation(names)

        assert result == [SanitizedName("unique1"), SanitizedName("unique2"), SanitizedName("unique3")]

    def test_mixed_collision_scenarios(self):
        """Test cases where some names collide and others don't in the same input list"""
        names = ["my-name", "my!name", "unique", "another-unique"]
        result = _clean_names_with_hash_disambiguation(names)

        # Should have 4 unique results
        assert len(set(result)) == 4

        # Unique names should remain simple
        assert SanitizedName("unique") in result
        assert SanitizedName("another_unique") in result

        # Colliding names should have hash suffixes
        colliding_results = [name for name in result if "my_name" in name]
        assert len(colliding_results) == 2
        assert all("_" in name and len(name) > len("my_name_") for name in colliding_results)


class TestHashReproducibility:
    """Test that hashing is reproducible"""

    def test_hash_reproducibility(self):
        """Test that the same input produces the same hash across multiple runs"""
        text = "test_string"
        hash1 = _get_short_hash(text)
        hash2 = _get_short_hash(text)

        assert hash1 == hash2

        # Test with the full collision resolution
        names = ["my-name", "my!name"]
        result1 = _clean_names_with_hash_disambiguation(names)
        result2 = _clean_names_with_hash_disambiguation(names)

        assert result1 == result2

    def test_hash_consistency_across_different_inputs(self):
        """Test that different inputs produce different hashes"""
        hash1 = _get_short_hash("input1")
        hash2 = _get_short_hash("input2")

        assert hash1 != hash2


class TestColumnMapper:
    """Test the ColumnMapper class"""

    def test_bidirectional_mapping(self):
        """Test that both forward and reverse lookups work correctly"""
        columns = ["User-Name", "Email@Address", "Phone!Number"]
        mapper = ColumnMapper(columns)

        # Test forward mapping
        assert "User-Name" in mapper.name_to_pg
        assert "Email@Address" in mapper.name_to_pg
        assert "Phone!Number" in mapper.name_to_pg

        # Test reverse mapping
        for original in columns:
            sanitized = mapper.name_to_pg[original]
            assert mapper.pg_to_name[sanitized] == original

        # Test that all mappings are unique
        assert len(mapper.name_to_pg) == len(columns)
        assert len(mapper.pg_to_name) == len(columns)

    def test_column_mapper_with_collisions(self):
        """Test ColumnMapper handles collisions correctly"""
        columns = ["my-name", "my!name", "unique_col"]
        mapper = ColumnMapper(columns)

        # Should have 3 unique sanitized names
        assert len(set(mapper.name_to_pg.values())) == 3

        # All original names should be mappable
        for col in columns:
            sanitized = mapper.name_to_pg[col]
            assert mapper.pg_to_name[sanitized] == col


class TestLargeInputHandling:
    """Test performance and correctness with larger datasets"""

    def test_large_input_handling(self):
        """Test performance and correctness with a larger list of names"""
        # Generate a mix of unique and colliding names
        names = [f"unique_name_{i}" for i in range(50)]

        # Add some colliding names
        for i in range(10):
            names.extend([f"collision-{i}", f"collision!{i}", f"collision${i}"])

        result = _clean_names_with_hash_disambiguation(names)

        # Should have same length
        assert len(result) == len(names)

        # All results should be unique
        assert len(set(result)) == len(names)

        # Unique names should not have hash suffixes
        unique_results = [name for name in result if name.startswith("unique_name_")]
        assert len(unique_results) == 50

        # All names should follow the expected pattern
        for name in result:
            assert re.match(r'^[a-z0-9_]+$', name), f"Invalid characters in: {name}"


class TestEdgeCases:
    """Test various edge cases"""

    def test_empty_input_list(self):
        """Test with empty input list"""
        result = _clean_names_with_hash_disambiguation([])
        assert result == []

    def test_single_item_list(self):
        """Test with single item list"""
        result = _clean_names_with_hash_disambiguation(["test-name"])
        assert result == [SanitizedName("test_name")]

    def test_all_identical_after_sanitization(self):
        """Test when all names become identical after sanitization"""
        names = ["@@@", "!!!", "$$$", "###"]
        result = _clean_names_with_hash_disambiguation(names)

        # Should have 4 unique results
        assert len(set(result)) == 4

        # All should be based on "_" but with different hashes
        for name in result:
            assert name.startswith("_")
            assert len(name) > 1  # Should have hash suffix
