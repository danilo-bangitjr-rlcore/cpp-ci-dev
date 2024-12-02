import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import MissingType, PipelineFrame, CallerCode
from corerl.data_pipeline.bound_checker import bound_checker
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.tag_config import TagConfig


def test_none_bounds():
    tag_1_cfg = TagConfig(name="tag_1", bounds=(None, None))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(None, None))

    data = pd.DataFrame({
        "tag_1": [3.4, -0.2, 2.7],
        "tag_2": [-0.4, 6.3, -3.8]
    })
    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)
    oob_mask = pf.missing_info == MissingType.BOUNDS.value

    assert not np.any(pf.data["tag_1"].isna())
    assert not np.any(pf.data["tag_2"].isna())
    assert not oob_mask.to_numpy().any()


def test_lower_bound_violation():
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 10.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [3.4, -0.2, 2.7, -0.5],
        "tag_2": [-0.4, 6.3, -3.8, 8.0]
    })
    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    assert np.all(
        np.isnan(pf.data["tag_1"]) == np.array([False, True, False, True])
    )
    assert np.all(
        np.isnan(pf.data["tag_2"]) == np.array([False, False, True, False])
    )
    assert pf.missing_info["tag_1"].iloc[1] == MissingType.BOUNDS
    assert pf.missing_info["tag_2"].iloc[2] == MissingType.BOUNDS


def test_upper_bound_violation():
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [0.4, 1.3, 0.7],
        "tag_2": [11.9, -0.5, 3.6]
    })
    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    assert np.all(
        np.isnan(pf.data["tag_1"]) == [False, True, False]
    )
    assert np.all(
        np.isnan(pf.data["tag_2"]) == [True, False, False]
    )
    assert pf.missing_info["tag_1"].iloc[1] == MissingType.BOUNDS
    assert pf.missing_info["tag_2"].iloc[0] == MissingType.BOUNDS


def test_multiple_missing_types():
    tag_1_cfg = TagConfig(name="tag_1", bounds=(0.0, 1.0))
    tag_2_cfg = TagConfig(name="tag_2", bounds=(-1.0, 10.0))

    data = pd.DataFrame({
        "tag_1": [np.nan, 1.3, 0.7],
        "tag_2": [11.9, -0.5, np.nan]
    })
    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = bound_checker(pf, 'tag_1', tag_1_cfg)
    pf = bound_checker(pf, 'tag_2', tag_2_cfg)

    assert np.all(
        np.isnan(pf.data["tag_1"]) == [True, True, False]
    )
    assert np.all(
        np.isnan(pf.data["tag_2"]) == [True, False, True]
    )
    assert pf.missing_info["tag_1"].iloc[0] == MissingType.MISSING
    assert pf.missing_info["tag_1"].iloc[1] == MissingType.BOUNDS
    assert pf.missing_info["tag_2"].iloc[0] == MissingType.BOUNDS
    assert pf.missing_info["tag_2"].iloc[2] == MissingType.MISSING
