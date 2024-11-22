import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.bound_checker import bound_checker
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.imputers.copy import CopyImputer, CopyImputerConfig
from corerl.data_pipeline.tag_config import TagConfig

def test_no_imputation():
    imputer_cfg_1 = CopyImputerConfig(default_val=0.0)
    imputer_cfg_2 = CopyImputerConfig(default_val=1.0)
    tag_1_imputer = CopyImputer(imputer_cfg_1)
    tag_2_imputer = CopyImputer(imputer_cfg_2)

    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [0.2, 0.3, 0.7],
        "tag_2": [1.9, -0.5, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    filtered_missing_pf = missing_data_checker(pf, 'tag_1')
    filtered_missing_pf = missing_data_checker(filtered_missing_pf, 'tag_2')

    filtered_bounds_pf = bound_checker(filtered_missing_pf, 'tag_1', tag_1_cfg)
    filtered_bounds_pf = bound_checker(filtered_bounds_pf, 'tag_2', tag_2_cfg)

    imputed_pf = tag_1_imputer(filtered_bounds_pf, 'tag_1')
    imputed_pf = tag_2_imputer(imputed_pf, 'tag_2')

    imputed_data = imputed_pf.data

    assert imputed_data["tag_1"].iloc[0] == 0.2
    assert imputed_data["tag_1"].iloc[1] == 0.3
    assert imputed_data["tag_1"].iloc[2] == 0.7
    assert imputed_data["tag_2"].iloc[0] == 1.9
    assert imputed_data["tag_2"].iloc[1] == -0.5
    assert imputed_data["tag_2"].iloc[2] == 7.3

def test_all_nan_imputation():
    imputer_cfg_1 = CopyImputerConfig(default_val=0.0)
    imputer_cfg_2 = CopyImputerConfig(default_val=1.0)
    tag_1_imputer = CopyImputer(imputer_cfg_1)
    tag_2_imputer = CopyImputer(imputer_cfg_2)

    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    filtered_missing_pf = missing_data_checker(pf, 'tag_1')
    filtered_missing_pf = missing_data_checker(filtered_missing_pf, 'tag_2')

    filtered_bounds_pf = bound_checker(filtered_missing_pf, 'tag_1', tag_1_cfg)
    filtered_bounds_pf = bound_checker(filtered_bounds_pf, 'tag_2', tag_2_cfg)

    imputed_pf = tag_1_imputer(filtered_bounds_pf, 'tag_1')
    imputed_pf = tag_2_imputer(imputed_pf, 'tag_2')

    imputed_data = imputed_pf.data

    assert imputed_data["tag_1"].iloc[0] == 0.0
    assert imputed_data["tag_1"].iloc[1] == 0.0
    assert imputed_data["tag_1"].iloc[2] == 0.0
    assert imputed_data["tag_2"].iloc[0] == 1.0
    assert imputed_data["tag_2"].iloc[1] == 1.0
    assert imputed_data["tag_2"].iloc[2] == 1.0

def test_nan_first_ind_imputation():
    imputer_cfg_1 = CopyImputerConfig(default_val=0.0)
    imputer_cfg_2 = CopyImputerConfig(default_val=1.0)
    tag_1_imputer = CopyImputer(imputer_cfg_1)
    tag_2_imputer = CopyImputer(imputer_cfg_2)

    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.3, 0.7],
        "tag_2": [-3.3, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    filtered_missing_pf = missing_data_checker(pf, 'tag_1')
    filtered_missing_pf = missing_data_checker(filtered_missing_pf, 'tag_2')

    filtered_bounds_pf = bound_checker(filtered_missing_pf, 'tag_1', tag_1_cfg)
    filtered_bounds_pf = bound_checker(filtered_bounds_pf, 'tag_2', tag_2_cfg)

    imputed_pf = tag_1_imputer(filtered_bounds_pf, 'tag_1')
    imputed_pf = tag_2_imputer(imputed_pf, 'tag_2')

    imputed_data = imputed_pf.data

    assert imputed_data["tag_1"].iloc[0] == 0.3
    assert imputed_data["tag_1"].iloc[1] == 0.3
    assert imputed_data["tag_1"].iloc[2] == 0.7
    assert imputed_data["tag_2"].iloc[0] == 7.3
    assert imputed_data["tag_2"].iloc[1] == 7.3
    assert imputed_data["tag_2"].iloc[2] == 7.3

def test_mixed_nan_imputation():
    imputer_cfg_1 = CopyImputerConfig(default_val=0.0)
    imputer_cfg_2 = CopyImputerConfig(default_val=1.0)
    tag_1_imputer = CopyImputer(imputer_cfg_1)
    tag_2_imputer = CopyImputer(imputer_cfg_2)

    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.3, np.nan],
        "tag_2": [-0.5, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data)

    filtered_missing_pf = missing_data_checker(pf, 'tag_1')
    filtered_missing_pf = missing_data_checker(filtered_missing_pf, 'tag_2')

    filtered_bounds_pf = bound_checker(filtered_missing_pf, 'tag_1', tag_1_cfg)
    filtered_bounds_pf = bound_checker(filtered_bounds_pf, 'tag_2', tag_2_cfg)

    imputed_pf = tag_1_imputer(filtered_bounds_pf, 'tag_1')
    imputed_pf = tag_2_imputer(imputed_pf, 'tag_2')

    imputed_data = imputed_pf.data

    assert imputed_data["tag_1"].iloc[0] == 0.3
    assert imputed_data["tag_1"].iloc[1] == 0.3
    assert imputed_data["tag_1"].iloc[2] == 0.3
    assert imputed_data["tag_2"].iloc[0] == -0.5
    assert imputed_data["tag_2"].iloc[1] == -0.5
    assert imputed_data["tag_2"].iloc[2] == 7.3
