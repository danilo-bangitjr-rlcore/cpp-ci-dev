import numpy as np
import pandas as pd
from lib_config.loader import direct_load_config
from test.infrastructure.utils.pandas import dfs_close

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.deltaize_tags import DeltaizeTags
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState


def test_one_delta_tag_one_pf():
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/test_delta_stage.yaml')
    assert isinstance(cfg, MainConfig)

    app_state = AppState(
        cfg=cfg,
        metrics=create_metrics_writer(cfg.metrics),
        evals=create_evals_writer(cfg.evals),
        event_bus=DummyEventBus(),
    )

    delta_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta, app_state)
    df = pd.DataFrame({
        "tag_1": [1, 1, 1, 1],
        "tag_2": [1, 2, 4, 3],
        "action_1": [0, 0, 0, 0],
        "reward": [1, 1, 1, 1],
    })
    dates = pd.DatetimeIndex(["7/13/2023 10:00", "7/13/2023 10:05", "7/13/2023 10:20", "7/13/2023 10:25"])
    df.index = dates

    pf = PipelineFrame(df, DataMode.ONLINE)
    out = delta_stage(pf)

    # Delta transform produces leading NaN if the temporal state was empty
    # Time threshold violation produces second NaN
    expected = pd.DataFrame({
        "tag_1": [1, 1, 1, 1],
        "tag_2": [np.nan, 1, np.nan, -1],
        "action_1": [0, 0, 0, 0],
        "reward": [1, 1, 1, 1],
    })
    expected.index = dates

    assert dfs_close(out.data, expected)

def test_one_delta_tag_multiple_pfs():
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/test_delta_stage.yaml')
    assert isinstance(cfg, MainConfig)

    app_state = AppState(
        cfg=cfg,
        metrics=create_metrics_writer(cfg.metrics),
        evals=create_evals_writer(cfg.evals),
        event_bus=DummyEventBus(),
    )
    pipeline = Pipeline(app_state, cfg.pipeline)

    df_1 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [1],
        "action_1": [0],
        "reward": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:00"]))

    got_1 = pipeline(
        df_1,
        data_mode=DataMode.ONLINE,
    )

    expected_1 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [np.nan],
        "action_1": [0],
        "action_1-lo": [0],
        "action_1-hi": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:00"]))

    assert dfs_close(got_1.df, expected_1)

    df_2 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [2],
        "action_1": [0],
        "reward": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:05"]))

    got_2 = pipeline(
        df_2,
        data_mode=DataMode.ONLINE,
    )

    # Entries are normalized
    expected_2 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [1],
        "action_1": [0],
        "action_1-lo": [0],
        "action_1-hi": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:05"]))

    assert dfs_close(got_2.df, expected_2)

    df_3 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [4],
        "action_1": [0],
        "reward": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:20"]))

    got_3 = pipeline(
        df_3,
        data_mode=DataMode.ONLINE,
    )

    expected_3 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [np.nan],
        "action_1": [0],
        "action_1-lo": [0],
        "action_1-hi": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:20"]))

    assert dfs_close(got_3.df, expected_3)

    df_4 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [3],
        "action_1": [0],
        "reward": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:25"]))

    got_4 = pipeline(
        df_4,
        data_mode=DataMode.ONLINE,
    )

    # Entries are normalized
    expected_4 = pd.DataFrame({
        "tag_1": [1],
        "tag_2": [0],
        "action_1": [0],
        "action_1-lo": [0],
        "action_1-hi": [1],
    }, index=pd.DatetimeIndex(["7/13/2023 10:25"]))

    assert dfs_close(got_4.df, expected_4)
