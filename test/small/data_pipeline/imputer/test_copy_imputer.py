import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, StageCode
from typing import cast, Dict
from corerl.data_pipeline.bound_checker import bound_checker
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.imputers.base import ImputerTemporalState
from corerl.data_pipeline.imputers.copy import CopyImputer, CopyImputerConfig
from corerl.data_pipeline.tag_config import TagConfig

def test_no_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [0.2, 0.3, 0.7],
        "tag_2": [1.9, -0.5, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create ImputerTemporalState
    tag_1_indices = pd.Series(["7/13/2023 9:00"])
    tag_1_indices = pd.to_datetime(tag_1_indices)
    tag_1_prev_pf_vals = pd.Series([0.9], name="tag_1", index=tag_1_indices)
    tag_2_indices = pd.Series(["7/13/2023 8:00", "7/13/2023 9:00"])
    tag_2_indices = pd.to_datetime(tag_2_indices)
    tag_2_prev_pf_vals = pd.Series([3.3, 4.5], name="tag_2", index=tag_2_indices)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )
    stage_ts["tag_1"] = ImputerTemporalState()
    stage_ts["tag_2"] = ImputerTemporalState()
    stage_ts["tag_1"].prev_pf_data = tag_1_prev_pf_vals
    stage_ts["tag_2"].prev_pf_data = tag_2_prev_pf_vals

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert pf.data["tag_1"].iloc[0] == 0.2
    assert pf.data["tag_1"].iloc[1] == 0.3
    assert pf.data["tag_1"].iloc[2] == 0.7
    assert pf.data["tag_2"].iloc[0] == 1.9
    assert pf.data["tag_2"].iloc[1] == -0.5
    assert pf.data["tag_2"].iloc[2] == 7.3
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert stage_ts["tag_1"].prev_pf_data.iloc[0] == 0.7
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert stage_ts["tag_2"].prev_pf_data.iloc[0] == -0.5
    assert stage_ts["tag_2"].prev_pf_data.iloc[1] == 7.3

def test_all_nan_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"]).all()
    assert np.isnan(pf.data["tag_2"]).all()
    assert isinstance(stage_ts["tag_1"].prev_pf_data, pd.Series | pd.DataFrame)
    assert isinstance(stage_ts["tag_2"].prev_pf_data, pd.Series | pd.DataFrame)
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert np.isnan(stage_ts["tag_1"].prev_pf_data.iloc[0])
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert np.isnan(stage_ts["tag_1"].prev_pf_data).all()

def test_all_nan_imputation_ts():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create ImputerTemporalState
    tag_1_indices = pd.Series(["7/13/2023 9:00"])
    tag_1_indices = pd.to_datetime(tag_1_indices)
    tag_1_prev_pf_vals = pd.Series([np.nan], name="tag_1", index=tag_1_indices)
    tag_2_indices = pd.Series(["7/13/2023 8:00", "7/13/2023 9:00"])
    tag_2_indices = pd.to_datetime(tag_2_indices)
    tag_2_prev_pf_vals = pd.Series([np.nan, np.nan], name="tag_2", index=tag_2_indices)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )
    stage_ts["tag_1"] = ImputerTemporalState()
    stage_ts["tag_2"] = ImputerTemporalState()
    stage_ts["tag_1"].prev_pf_data = tag_1_prev_pf_vals
    stage_ts["tag_2"].prev_pf_data = tag_2_prev_pf_vals

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"]).all()
    assert np.isnan(pf.data["tag_2"]).all()
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert np.isnan(stage_ts["tag_1"].prev_pf_data.iloc[0])
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert np.isnan(stage_ts["tag_1"].prev_pf_data).all()

def test_backtrack_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [0.1, 0.7, np.nan],
        "tag_2": [-0.3, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert pf.data["tag_1"].iloc[0] == 0.1
    assert pf.data["tag_1"].iloc[1] == 0.7
    assert pf.data["tag_1"].iloc[2] == 0.7
    assert pf.data["tag_2"].iloc[0] == -0.3
    assert pf.data["tag_2"].iloc[1] == -0.3
    assert pf.data["tag_2"].iloc[2] == 7.3
    assert isinstance(stage_ts["tag_1"].prev_pf_data, pd.Series | pd.DataFrame)
    assert isinstance(stage_ts["tag_2"].prev_pf_data, pd.Series | pd.DataFrame)
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert np.isnan(stage_ts["tag_1"].prev_pf_data.iloc[0])
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert np.isnan(stage_ts["tag_2"].prev_pf_data.iloc[0])
    assert stage_ts["tag_2"].prev_pf_data.iloc[1] == 7.3

def test_backtrack_imputation_ts():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, np.nan],
        "tag_2": [np.nan, 6.6, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create ImputerTemporalState
    tag_1_indices = pd.Series(["7/13/2023 9:00"])
    tag_1_indices = pd.to_datetime(tag_1_indices)
    tag_1_prev_pf_vals = pd.Series([0.4], name="tag_1", index=tag_1_indices)
    tag_2_indices = pd.Series(["7/13/2023 8:00", "7/13/2023 9:00"])
    tag_2_indices = pd.to_datetime(tag_2_indices)
    tag_2_prev_pf_vals = pd.Series([0.6, np.nan], name="tag_2", index=tag_2_indices)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )
    stage_ts["tag_1"] = ImputerTemporalState()
    stage_ts["tag_2"] = ImputerTemporalState()
    stage_ts["tag_1"].prev_pf_data = tag_1_prev_pf_vals
    stage_ts["tag_2"].prev_pf_data = tag_2_prev_pf_vals

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert pf.data["tag_1"].iloc[0] == 0.4
    assert pf.data["tag_1"].iloc[1] == 0.7
    assert pf.data["tag_1"].iloc[2] == 0.7
    assert pf.data["tag_2"].iloc[0] == 0.6
    assert pf.data["tag_2"].iloc[1] == 6.6
    assert pf.data["tag_2"].iloc[2] == 6.6
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert np.isnan(stage_ts["tag_1"].prev_pf_data.iloc[0])
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert stage_ts["tag_2"].prev_pf_data.iloc[0] == 6.6
    assert np.isnan(stage_ts["tag_2"].prev_pf_data.iloc[1])

def test_lookahead_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, 0.1],
        "tag_2": [np.nan, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"].iloc[0])
    assert pf.data["tag_1"].iloc[1] == 0.1
    assert pf.data["tag_1"].iloc[2] == 0.1
    assert pf.data["tag_2"].iloc[0] == 7.3
    assert pf.data["tag_2"].iloc[1] == 7.3
    assert pf.data["tag_2"].iloc[2] == 7.3
    assert isinstance(stage_ts["tag_1"].prev_pf_data, pd.Series | pd.DataFrame)
    assert isinstance(stage_ts["tag_2"].prev_pf_data, pd.Series | pd.DataFrame)
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert stage_ts["tag_1"].prev_pf_data.iloc[0] == 0.1
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert np.isnan(stage_ts["tag_2"].prev_pf_data.iloc[0])
    assert stage_ts["tag_2"].prev_pf_data.iloc[1] == 7.3

def test_lookahead_imputation_ts():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, 0.1],
        "tag_2": [np.nan, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create ImputerTemporalState
    tag_1_indices = pd.Series(["7/13/2023 9:00"])
    tag_1_indices = pd.to_datetime(tag_1_indices)
    tag_1_prev_pf_vals = pd.Series([np.nan], name="tag_1", index=tag_1_indices)
    tag_2_indices = pd.Series(["7/13/2023 8:00", "7/13/2023 9:00"])
    tag_2_indices = pd.to_datetime(tag_2_indices)
    tag_2_prev_pf_vals = pd.Series([np.nan, np.nan], name="tag_2", index=tag_2_indices)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )
    stage_ts["tag_1"] = ImputerTemporalState()
    stage_ts["tag_2"] = ImputerTemporalState()
    stage_ts["tag_1"].prev_pf_data = tag_1_prev_pf_vals
    stage_ts["tag_2"].prev_pf_data = tag_2_prev_pf_vals

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"].iloc[0])
    assert pf.data["tag_1"].iloc[1] == 0.1
    assert pf.data["tag_1"].iloc[2] == 0.1
    assert pf.data["tag_2"].iloc[0] == 7.3
    assert pf.data["tag_2"].iloc[1] == 7.3
    assert pf.data["tag_2"].iloc[2] == 7.3
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert stage_ts["tag_1"].prev_pf_data.iloc[0] == 0.1
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert np.isnan(stage_ts["tag_2"].prev_pf_data.iloc[0])
    assert stage_ts["tag_2"].prev_pf_data.iloc[1] == 7.3

def test_mixed_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.1, np.nan, 0.4],
        "tag_2": [np.nan, np.nan, 7.3, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00", "7/13/2023 13:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert pf.data["tag_1"].iloc[0] == 0.1
    assert pf.data["tag_1"].iloc[1] == 0.1
    assert pf.data["tag_1"].iloc[2] == 0.1
    assert pf.data["tag_1"].iloc[3] == 0.4
    assert pf.data["tag_2"].iloc[0] == 7.3
    assert pf.data["tag_2"].iloc[1] == 7.3
    assert pf.data["tag_2"].iloc[2] == 7.3
    assert pf.data["tag_2"].iloc[3] == 7.3
    assert isinstance(stage_ts["tag_1"].prev_pf_data, pd.Series | pd.DataFrame)
    assert isinstance(stage_ts["tag_2"].prev_pf_data, pd.Series | pd.DataFrame)
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert stage_ts["tag_1"].prev_pf_data.iloc[0] == 0.4
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert stage_ts["tag_2"].prev_pf_data.iloc[0] == 7.3
    assert np.isnan(stage_ts["tag_2"].prev_pf_data.iloc[1])

def test_mixed_imputation_ts():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = CopyImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = CopyImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = CopyImputer(imputer_cfg_1, tag_1_cfg)
    tag_2_imputer = CopyImputer(imputer_cfg_2, tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.1, np.nan, 0.4],
        "tag_2": [np.nan, np.nan, 7.3, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00", "7/13/2023 13:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create ImputerTemporalState
    tag_1_indices = pd.Series(["7/13/2023 9:00"])
    tag_1_indices = pd.to_datetime(tag_1_indices)
    tag_1_prev_pf_vals = pd.Series([np.nan], name="tag_1", index=tag_1_indices)
    tag_2_indices = pd.Series(["7/13/2023 8:00", "7/13/2023 9:00"])
    tag_2_indices = pd.to_datetime(tag_2_indices)
    tag_2_prev_pf_vals = pd.Series([6.8, np.nan], name="tag_2", index=tag_2_indices)

    pf.temporal_state[StageCode.IMPUTER] = dict()
    stage_ts = pf.temporal_state[StageCode.IMPUTER]
    assert isinstance(stage_ts, dict)
    stage_ts = cast(
        Dict[str, ImputerTemporalState],
        stage_ts
    )
    stage_ts["tag_1"] = ImputerTemporalState()
    stage_ts["tag_2"] = ImputerTemporalState()
    stage_ts["tag_1"].prev_pf_data = tag_1_prev_pf_vals
    stage_ts["tag_2"].prev_pf_data = tag_2_prev_pf_vals

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert pf.data["tag_1"].iloc[0] == 0.1
    assert pf.data["tag_1"].iloc[1] == 0.1
    assert pf.data["tag_1"].iloc[2] == 0.1
    assert pf.data["tag_1"].iloc[3] == 0.4
    assert pf.data["tag_2"].iloc[0] == 6.8
    assert pf.data["tag_2"].iloc[1] == 7.3
    assert pf.data["tag_2"].iloc[2] == 7.3
    assert pf.data["tag_2"].iloc[3] == 7.3
    assert len(stage_ts["tag_1"].prev_pf_data) == 1
    assert stage_ts["tag_1"].prev_pf_data.iloc[0] == 0.4
    assert len(stage_ts["tag_2"].prev_pf_data) == 2
    assert stage_ts["tag_2"].prev_pf_data.iloc[0] == 7.3
    assert np.isnan(stage_ts["tag_2"].prev_pf_data.iloc[1])
