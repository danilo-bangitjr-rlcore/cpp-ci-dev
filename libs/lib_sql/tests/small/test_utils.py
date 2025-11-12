from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lib_sql.utils import (
    ColumnMapper,
    SanitizedName,
    SQLColumn,
    _clean_names_with_hash_disambiguation,
    _get_short_hash,
    _sanitize_key,
    add_column_to_table_query,
    create_sqlite_table_query,
    create_tsdb_table_query,
    sanitize_keys,
)


@pytest.fixture
def db_connection(tmp_path: Path):
    """Create a database connection that automatically closes."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()


class TestCreateSqliteTableQuery:
    @pytest.mark.timeout(5)
    def test_create_sqlite_table_with_real_database(self, db_connection: sqlite3.Connection):
        """
        Test SQLite table creation by actually creating the table in a real database.
        """
        columns = [
            SQLColumn(name="id", type="INTEGER", nullable=False),
            SQLColumn(name="data", type="TEXT", nullable=True),
        ]

        result = create_sqlite_table_query(
            schema="ignored",
            table="test_table",
            columns=columns,
            index_columns=["id"],
        )

        cursor = db_connection.cursor()
        cursor.executescript(str(result))

        cursor.execute("PRAGMA table_info(test_table)")
        table_info = cursor.fetchall()

        columns_info = {row[1]: {"type": row[2], "not_null": bool(row[3])} for row in table_info}

        assert "id" in columns_info
        assert "data" in columns_info
        assert columns_info["id"]["not_null"] is True
        assert columns_info["data"]["not_null"] is False

        cursor.execute("PRAGMA index_list(test_table)")
        indexes = cursor.fetchall()
        index_names = [idx[1] for idx in indexes]
        assert any("test_table_id_idx" in name for name in index_names)


class TestSanitizeKeyAdvanced:
    @pytest.mark.timeout(5)
    @pytest.mark.parametrize(
        "input_key,expected",
        [
            ("用户数据", "_"),
            ("données_utilisateur", "donn_es_utilisateur"),
            ("пользователь", "_"),
            ("🔥_热门", "_"),
            ("'; DROP TABLE users; --", "_drop_table_users_"),
            ("OR 1=1", "or_1_1"),
            ("___", "_"),
            ("A_B__C___D", "a_b_c_d"),
            ("123numbers", "123numbers"),
        ],
    )
    def test_sanitize_key_security_and_unicode(self, input_key: str, expected: str):
        """
        Test key sanitization handles security issues and international characters.
        """
        assert _sanitize_key(input_key) == expected


class TestColumnMapperAdvanced:
    @pytest.mark.timeout(5)
    def test_column_mapper_large_scale_collision(self):
        """
        Test ColumnMapper handles large-scale duplicate name scenarios.
        """
        columns = [
            "User-Data",
            "user_data",
            "User.Data",
            "USER_DATA",
            "user__data",
            "User Data",
            "user@data",
            "user#data",
            "user$data",
            "user%data",
        ]

        mapper = ColumnMapper(columns)

        sanitized_names = list(mapper.name_to_pg.values())
        assert len(set(sanitized_names)) == len(columns)

        for original in columns:
            sanitized = mapper.name_to_pg[original]
            assert mapper.pg_to_name[sanitized] == original


class TestIntegrationWorkflows:
    @pytest.mark.timeout(5)
    def test_sqlite_table_creation_integration(self, db_connection: sqlite3.Connection):
        """
        Test SQLite table creation with column mapping integration.
        """
        raw_columns = [
            SQLColumn(name="Time-Stamp", type="TIMESTAMP", nullable=False),
            SQLColumn(name="Device.ID", type="INTEGER", nullable=False),
            SQLColumn(name="Sensor Value", type="DECIMAL(10,2)", nullable=True),
        ]

        column_names = [col.name for col in raw_columns]
        mapper = ColumnMapper(column_names)

        mapped_columns = [
            SQLColumn(
                name=mapper.name_to_pg[col.name],
                type=col.type,
                nullable=col.nullable,
            )
            for col in raw_columns
        ]

        result = create_sqlite_table_query(
            schema="test",
            table="mapped_table",
            columns=mapped_columns,
            index_columns=[mapper.name_to_pg["Device.ID"]],
        )

        cursor = db_connection.cursor()
        cursor.executescript(str(result))

        cursor.execute("PRAGMA table_info(mapped_table)")
        table_info = cursor.fetchall()
        column_names_in_db = [row[1] for row in table_info]

        assert "time_stamp" in column_names_in_db
        assert "device_id" in column_names_in_db
        assert "sensor_value" in column_names_in_db

        assert "Time-Stamp" not in column_names_in_db
        assert "Device.ID" not in column_names_in_db
        assert "Sensor Value" not in column_names_in_db

    @pytest.mark.timeout(5)
    def test_empty_columns_edge_case(self):
        """
        Test table creation with empty column list generates expected SQL structure.
        """
        result = create_sqlite_table_query(schema="test", table="empty_table", columns=[])

        query_str = str(result)
        assert "CREATE TABLE IF NOT EXISTS empty_table" in query_str


class TestAddColumnToTableQuery:
    @pytest.mark.timeout(5)
    @pytest.mark.parametrize(
        "nullable,expected_null",
        [
            (True, ""),
            (False, "NOT NULL"),
        ],
    )
    def test_add_column_nullable_handling(self, nullable: bool, expected_null: str):
        """
        Test column addition with nullable/non-nullable handling.
        """
        column = SQLColumn(name="new_field", type="VARCHAR(255)", nullable=nullable)

        result = add_column_to_table_query("test", "table", column)
        query_str = str(result)

        assert "ALTER TABLE test.table" in query_str
        assert "ADD COLUMN new_field VARCHAR(255)" in query_str
        if expected_null:
            assert expected_null in query_str
        else:
            assert "NOT NULL" not in query_str


class TestSanitizeKey:
    @pytest.mark.timeout(5)
    @pytest.mark.parametrize(
        "input_key,expected",
        [
            ("simple_key", "simple_key"),
            ("SimpleKey", "simplekey"),
            ("key-with-dashes", "key_with_dashes"),
            ("key.with.dots", "key_with_dots"),
            ("key with spaces", "key_with_spaces"),
            ("key@special!chars", "key_special_chars"),
            ("key___multiple___underscores", "key_multiple_underscores"),
            ("用户数据", "_"),
            ("", ""),
        ],
    )
    def test_sanitize_key_transformations(self, input_key: str, expected: str):
        """
        Test key sanitization with various input patterns.
        """
        assert _sanitize_key(input_key) == expected


class TestGetShortHash:
    @pytest.mark.timeout(5)
    @pytest.mark.parametrize("length", [1, 2, 4, 8])
    def test_get_short_hash_custom_length(self, length: int):
        """
        Test hash generation with custom lengths.
        """
        hash_result = _get_short_hash("test", length)
        assert len(hash_result) == length
        assert all(c in "0123456789abcdef" for c in hash_result)


class TestCleanNamesWithHashDisambiguation:
    @pytest.mark.timeout(5)
    def test_clean_names_no_duplicates(self):
        """
        Test cleaning unique names preserves them.
        """
        names = ["simple", "another", "third"]
        result = _clean_names_with_hash_disambiguation(names)

        assert result == [SanitizedName("simple"), SanitizedName("another"), SanitizedName("third")]

    @pytest.mark.timeout(5)
    def test_clean_names_with_duplicates(self):
        """
        Test cleaning names with duplicates adds hash disambiguation.
        """
        names = ["User-Data", "user_data", "User.Data"]
        result = _clean_names_with_hash_disambiguation(names)

        assert len(result) == 3
        assert len(set(result)) == 3

        for name in result:
            assert name.startswith("user_data_")
            assert len(name) == len("user_data_") + 4

    @pytest.mark.timeout(5)
    def test_clean_names_deterministic(self):
        """
        Test that cleaning is deterministic.
        """
        names = ["Same-Name", "same_name", "Same.Name"]

        result1 = _clean_names_with_hash_disambiguation(names)
        result2 = _clean_names_with_hash_disambiguation(names)

        assert result1 == result2


class TestColumnMapper:
    @pytest.mark.timeout(5)
    def test_column_mapper_with_duplicates(self):
        """
        Test ColumnMapper handles duplicate names after sanitization.
        """
        columns = ["User-Data", "user_data", "User.Data"]
        mapper = ColumnMapper(columns)

        assert len(set(mapper.name_to_pg.values())) == 3

        for sanitized in mapper.name_to_pg.values():
            assert sanitized.startswith("user_data_")


class TestSanitizeKeys:
    @pytest.mark.timeout(5)
    def test_sanitize_keys_transforms_preserve_values(self):
        """
        Test key sanitization transforms keys while preserving values.
        """
        dict_points = [
            {"User-Name": "john", "User.Email": "john@example.com", "Age": 30},
            {"User-Name": "jane", "User.Email": "jane@example.com", "Age": 25},
        ]

        result = sanitize_keys(dict_points)

        assert len(result) == 2

        for point in result:
            assert "user_name" in point
            assert "user_email" in point
            assert "age" in point
            assert "User-Name" not in point
            assert "User.Email" not in point

        assert result[0]["user_name"] == "john"
        assert result[0]["age"] == 30
        assert result[1]["user_name"] == "jane"
        assert result[1]["age"] == 25


class TestUtilsIntegration:
    @pytest.mark.timeout(5)
    def test_full_table_creation_workflow(self):
        """
        Test complete workflow from raw columns to sanitized table creation.
        """
        raw_columns = [
            SQLColumn(name="Time-Stamp", type="TIMESTAMP WITH TIME ZONE", nullable=False),
            SQLColumn(name="Device.ID", type="INTEGER", nullable=False),
            SQLColumn(name="Sensor Value", type="DECIMAL(10,2)", nullable=True),
        ]

        column_names = [col.name for col in raw_columns]
        mapper = ColumnMapper(column_names)

        mapped_columns = [
            SQLColumn(
                name=mapper.name_to_pg[col.name],
                type=col.type,
                nullable=col.nullable,
            )
            for col in raw_columns
        ]

        result = create_tsdb_table_query(
            schema="sensors",
            table="readings",
            columns=mapped_columns,
            partition_column=mapper.name_to_pg["Device.ID"],
            index_columns=[mapper.name_to_pg["Device.ID"]],
            time_column=mapper.name_to_pg["Time-Stamp"],
        )

        query_str = str(result)

        assert "time_stamp" in query_str
        assert "device_id" in query_str
        assert "sensor_value" in query_str

        assert "Time-Stamp" not in query_str
        assert "Device.ID" not in query_str
        assert "Sensor Value" not in query_str
