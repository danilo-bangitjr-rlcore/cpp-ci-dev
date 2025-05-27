import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, Tuple

import pytest

from corerl.bin.convert_db_backup import convert_table


@pytest.fixture
def mock_csv_file() -> Generator[Tuple[Path, Path], None, None]:
    """Creates a temporary CSV file for testing."""
    with TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        output_path = Path(tmpdir) / "output.csv"

        # Write mock data to input CSV
        with open(input_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Header1", "Header2", "Header3", "time", "Tag1", "Tag2", "Tag3"])
            writer.writerow(["", "", "", "", "Boolean", "Integer", "Float"])
            writer.writerow(["", "", "", "time", "Tag1", "Tag2", "Tag3"])
            writer.writerow(["", "", "", "2025-01-01T00:00:00", "true", "42", "3.14"])
            writer.writerow(["", "", "", "2025-01-01T00:00:01", "false", "43", "3.15"])

        yield input_path, output_path


def test_convert_table(mock_csv_file: Tuple[Path, Path]):
    """Tests the convert_table function."""
    input_path, output_path = mock_csv_file

    convert_table(
        path_file=input_path,
        host="TestHost",
        quality="TestQuality",
        output=output_path,
        row_offset=3,
        tag_name_row=2
    )

    # Validate output
    assert output_path.exists(), "Output file was not created."

    with open(output_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check headers
    processed_columns = ["time", "host", "id", "name", "Quality", "fields"]
    assert rows[0] == processed_columns, "Output headers do not match."

    # Check first row of data
    expected_data_row = [
        "2025-01-01T00:00:00",
        "TestHost",
        "ns=2;s=Tag1",
        "Tag1",
        "TestQuality",
        "{'val': True, 'DataType': 'Boolean'}"
    ]
    assert rows[1] == expected_data_row, "Output data does not match expected values."
    assert "'val': True" in rows[1][5], "Boolean value was not processed correctly."

    expected_data_row_2 = [
        "2025-01-01T00:00:00",
        "TestHost",
        "ns=2;s=Tag2",
        "Tag2",
        "TestQuality",
        "{'val': 42, 'DataType': 'Integer'}"
    ]
    assert rows[2] == expected_data_row_2, "Output data does not match expected integer values."

    expected_data_row_3 = [
        "2025-01-01T00:00:00",
        "TestHost",
        "ns=2;s=Tag3",
        "Tag3",
        "TestQuality",
        "{'val': 3.14, 'DataType': 'Float'}"
    ]
    assert rows[3] == expected_data_row_3, "Output data does not match expected float values."


if __name__ == "__main__":
    pytest.main()
