from corerl.data_loaders.factory import init_data_loader
from corerl.data_loaders.direct_action import DirectActionDataLoaderConfig, DirectActionDataLoader


def test_direct_action_data_loader1():
    """
    Calling the init_data_loader_new factory function
    with a DirectActionDataLoaderConfig gives back
    a DirectActionDataLoader.
    """
    cfg = DirectActionDataLoaderConfig(
        offline_data_path="offline_data/",
        train_filenames=["file1.csv"],
        test_filenames=["file2.csv"],
        skip_rows=0,
        header=0,
        df_col_names=["Date", "Action", "Sensor_1", "Sensor_2"],
        obs_col_names=["Sensor_1", "Sensor_2"],
        action_col_names=["Action"],
        date_col_name='Date',
        max_time_delta=30,
        obs_length=60,
        steps_per_decision=5,
        only_dp_transitions=False
    )

    dl = init_data_loader(cfg)

    assert isinstance(dl, DirectActionDataLoader)
