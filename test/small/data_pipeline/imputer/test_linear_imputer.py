import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.bound_checker import bound_checker
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.imputers.linear import LinearImputer, LinearImputerConfig
from corerl.data_pipeline.tag_config import TagConfig

def test_no_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = LinearImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = LinearImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = LinearImputer(imputer_cfg_1, tag_cfg=tag_1_cfg)
    tag_2_imputer = LinearImputer(imputer_cfg_2, tag_cfg=tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [0.0, 0.2, 0.4],
        "tag_2": [10.0, 8.0, 6.0]
    })
    indices = pd.Series(["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert pf.data["tag_1"].iloc[0] == 0.0
    assert pf.data["tag_1"].iloc[1] == 0.2
    assert pf.data["tag_1"].iloc[2] == 0.4
    assert pf.data["tag_2"].iloc[0] == 10.0
    assert pf.data["tag_2"].iloc[1] == 8.0
    assert pf.data["tag_2"].iloc[2] == 6.0

def test_all_nan_imputation():
    tag_1_horizon = 1
    tag_2_horizon = 2
    imputer_cfg_1 = LinearImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = LinearImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = LinearImputer(imputer_cfg_1, tag_cfg=tag_1_cfg)
    tag_2_imputer = LinearImputer(imputer_cfg_2, tag_cfg=tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"]).all()
    assert np.isnan(pf.data["tag_2"]).all()

def test_only_nan_in_imputation_horizon():
    tag_1_horizon = 2
    tag_2_horizon = 2
    imputer_cfg_1 = LinearImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = LinearImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = LinearImputer(imputer_cfg_1, tag_cfg=tag_1_cfg)
    tag_2_imputer = LinearImputer(imputer_cfg_2, tag_cfg=tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan, 0.6, 0.8, 1.0],
        "tag_2": [10.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    indices = pd.Series(
        ["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00", "7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"].iloc[0])
    assert np.isclose(pf.data["tag_1"].iloc[1], 0.6)
    assert np.isclose(pf.data["tag_1"].iloc[2], 0.4)
    assert np.isclose(pf.data["tag_1"].iloc[3], 0.6)
    assert np.isclose(pf.data["tag_1"].iloc[4], 0.8)
    assert np.isclose(pf.data["tag_1"].iloc[5], 1.0)
    assert np.isclose(pf.data["tag_2"].iloc[0], 10.0)
    assert np.isclose(pf.data["tag_2"].iloc[1], 10.0)
    assert np.isclose(pf.data["tag_2"].iloc[2], 10.0)
    assert np.isnan(pf.data["tag_2"].iloc[3])
    assert np.isnan(pf.data["tag_2"].iloc[4])
    assert np.isnan(pf.data["tag_2"].iloc[5])

def test_backtrack_imputation():
    tag_1_horizon = 2
    tag_2_horizon = 2
    imputer_cfg_1 = LinearImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = LinearImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = LinearImputer(imputer_cfg_1, tag_cfg=tag_1_cfg)
    tag_2_imputer = LinearImputer(imputer_cfg_2, tag_cfg=tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [0.0, 0.2, 0.4, 0.6, 0.9, np.nan],
        "tag_2": [10.0, 8.0, 6.0, 4.0, np.nan, np.nan]
    })
    indices = pd.Series(
        ["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00", "7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isclose(pf.data["tag_1"].iloc[0], 0.0)
    assert np.isclose(pf.data["tag_1"].iloc[1], 0.2)
    assert np.isclose(pf.data["tag_1"].iloc[2], 0.4)
    assert np.isclose(pf.data["tag_1"].iloc[3], 0.6)
    assert np.isclose(pf.data["tag_1"].iloc[4], 0.9)
    assert np.isclose(pf.data["tag_1"].iloc[5], 1.0)
    assert np.isclose(pf.data["tag_2"].iloc[0], 10.0)
    assert np.isclose(pf.data["tag_2"].iloc[1], 8.0)
    assert np.isclose(pf.data["tag_2"].iloc[2], 6.0)
    assert np.isclose(pf.data["tag_2"].iloc[3], 4.0)
    assert np.isclose(pf.data["tag_2"].iloc[4], 2.0)
    assert np.isclose(pf.data["tag_2"].iloc[5], 4.0)

def test_lookahead_imputation():
    tag_1_horizon = 2
    tag_2_horizon = 2
    imputer_cfg_1 = LinearImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = LinearImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = LinearImputer(imputer_cfg_1, tag_cfg=tag_1_cfg)
    tag_2_imputer = LinearImputer(imputer_cfg_2, tag_cfg=tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.1, 0.4, 0.6, 0.8, 1.0],
        "tag_2": [np.nan, np.nan, 6.0, 4.0, 2.0, 0.0]
    })
    indices = pd.Series(
        ["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00", "7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isclose(pf.data["tag_1"].iloc[0], 0.0)
    assert np.isclose(pf.data["tag_1"].iloc[1], 0.1)
    assert np.isclose(pf.data["tag_1"].iloc[2], 0.4)
    assert np.isclose(pf.data["tag_1"].iloc[3], 0.6)
    assert np.isclose(pf.data["tag_1"].iloc[4], 0.8)
    assert np.isclose(pf.data["tag_1"].iloc[5], 1.0)
    assert np.isclose(pf.data["tag_2"].iloc[0], 6.0)
    assert np.isclose(pf.data["tag_2"].iloc[1], 8.0)
    assert np.isclose(pf.data["tag_2"].iloc[2], 6.0)
    assert np.isclose(pf.data["tag_2"].iloc[3], 4.0)
    assert np.isclose(pf.data["tag_2"].iloc[4], 2.0)
    assert np.isclose(pf.data["tag_2"].iloc[5], 0.0)

def test_straddle_imputation():
    tag_1_horizon = 2
    tag_2_horizon = 2
    imputer_cfg_1 = LinearImputerConfig(imputation_horizon=tag_1_horizon)
    imputer_cfg_2 = LinearImputerConfig(imputation_horizon=tag_2_horizon)
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))
    tag_1_imputer = LinearImputer(imputer_cfg_1, tag_cfg=tag_1_cfg)
    tag_2_imputer = LinearImputer(imputer_cfg_2, tag_cfg=tag_2_cfg)

    data = pd.DataFrame({
        "tag_1": [0.0, np.nan, 0.4, 0.6, np.nan, 1.0],
        "tag_2": [10.0, 8.0, np.nan, np.nan, 2.0, 0.0]
    })
    indices = pd.Series(
        ["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00", "7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isclose(pf.data["tag_1"].iloc[0], 0.0)
    assert np.isclose(pf.data["tag_1"].iloc[1], 0.2)
    assert np.isclose(pf.data["tag_1"].iloc[2], 0.4)
    assert np.isclose(pf.data["tag_1"].iloc[3], 0.6)
    assert np.isclose(pf.data["tag_1"].iloc[4], 0.8)
    assert np.isclose(pf.data["tag_1"].iloc[5], 1.0)
    assert np.isclose(pf.data["tag_2"].iloc[0], 10.0)
    assert np.isclose(pf.data["tag_2"].iloc[1], 8.0)
    assert np.isclose(pf.data["tag_2"].iloc[2], 6.0)
    assert np.isclose(pf.data["tag_2"].iloc[3], 4.0)
    assert np.isclose(pf.data["tag_2"].iloc[4], 2.0)
    assert np.isclose(pf.data["tag_2"].iloc[5], 0.0)
