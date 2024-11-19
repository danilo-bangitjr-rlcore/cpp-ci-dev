import pandas as pd

from corerl.data.data import MissingType, SparseMissingType


def test_memory_usage():
    N = 1_000_000
    missing_data = {
        "sensor_x": [MissingType.NULL] * N,
        "sensor_y": [MissingType.NULL] * N,
    }
    # add some missing data flags
    missing_data["sensor_x"][1234] = MissingType.BOUNDS
    missing_data["sensor_x"][1235] = MissingType.OUTLIER
    missing_data["sensor_y"][1236] = MissingType.BOUNDS
    missing_data["sensor_y"][1237] = MissingType.OUTLIER

    # create sparse df and nonsparse df to compare
    sparse_df = pd.DataFrame(missing_data, dtype=SparseMissingType)
    non_sparse_df = pd.DataFrame(missing_data, dtype=int)

    # get memory in bytes
    sparse_mem = sparse_df.memory_usage().sum() / 1e3
    non_sparse_mem = non_sparse_df.memory_usage().sum() / 1e3

    # you can run this file as a script to see some output
    print(f"{sparse_mem=} bytes")
    print(f"{non_sparse_mem=} bytes")
    print(f"ratio: {non_sparse_mem/sparse_mem}")
    print(sparse_df.iloc[1234:1238])

    # showcase some usage of IntFlag enum
    assert sparse_df["sensor_x"].iloc[1234] == MissingType.BOUNDS

    missing_type_x = MissingType(sparse_df["sensor_x"].iloc[1234])
    assert missing_type_x in MissingType.BOUNDS | MissingType.OUTLIER

    assert sparse_mem < 0.2  # less than 0.2 bytes


if __name__ == "__main__":
    test_memory_usage()
