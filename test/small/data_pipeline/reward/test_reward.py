import datetime
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

import corerl.data_pipeline.transforms as xform
from corerl.data_pipeline.constructors.rc import RewardComponentConstructor, RewardConstructor
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from test.infrastructure.utils.pandas import dfs_close


def test_rc1():
    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 2, 3, np.nan, 1, 2],
            "obs_2": [1, 2, 3, np.nan, 1, 2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    transform_cfgs = [
        xform.NormalizerConfig(min=0.0, max=1.0, from_data=False),
        xform.TraceConfig(trace_values=[0.1]),
    ]
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs) for tag_name in raw_obs.columns
    }
    rc = RewardConstructor(reward_component_constructors)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs_1_trace-0.1": [np.nan, 1.0, 1.9, 2.89, np.nan, 1.0, 1.9],
            "obs_2_trace-0.1": [1.0, 1.9, 2.89, np.nan, 1.0, 1.9, np.nan],
        }
    )
    expected_reward_vals = expected_components.sum(axis=1, skipna=False)
    expected_reward_df = pd.DataFrame({"reward": expected_reward_vals})
    expected_df = pd.concat((raw_obs, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_null_xform():
    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 2, 3, np.nan, 1, 2],
            "obs_2": [1, 2, 3, np.nan, 1, 2, np.nan],
            "obs_3": [1, 2, 3, np.nan, 1, 2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    transform_cfgs = [
        xform.NormalizerConfig(min=0.0, max=1.0, from_data=False),
        xform.TraceConfig(trace_values=[0.1]),
    ]
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs) for tag_name in raw_obs.columns
    }
    # change final xform to null
    reward_component_constructors["obs_3"] = RewardComponentConstructor([xform.NullConfig()])
    rc = RewardConstructor(reward_component_constructors)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs_1_trace-0.1": [np.nan, 1.0, 1.9, 2.89, np.nan, 1.0, 1.9],
            "obs_2_trace-0.1": [1.0, 1.9, 2.89, np.nan, 1.0, 1.9, np.nan],
        }
    )
    expected_reward_vals = expected_components.sum(axis=1, skipna=False)
    expected_reward_df = pd.DataFrame({"reward": expected_reward_vals})
    expected_df = pd.concat((raw_obs, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_lessthan_xform():
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xform.LessThanConfig(threshold=5),
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([True, True, True, False, np.nan, False, False])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)
    expected_df = pd.concat((df, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_greaterthan_xform():
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xform.GreaterThanConfig(threshold=5),
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([False, False, False, True, np.nan, True, True])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)
    expected_df = pd.concat((df, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_greaterthan_penalty_reward():
    """
    This tests the following reward:
        if tag_2 > 5:
            r = -10
        else:
            r = 0
    """
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xform.GreaterThanConfig(threshold=5),
                xform.ScaleConfig(factor=-10) # penalty
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([0.0, 0.0, 0.0, -10.0, np.nan, -10.0, -10.0])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)
    expected_df = pd.concat((df, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_product_transform():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0,      2],
            [1,      4],
            [2,      6],
            [np.nan, np.nan],
            [4,      10],
            [5,      12],
        ],
        columns=cols,
        index=idx,
    )

    transform_cfgs = {
        "tag-1": [xform.ProductConfig(other="tag-2", other_xform=[xform.GreaterThanConfig(threshold=5)])],
        "tag-2": [xform.NullConfig()],
    }
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs[tag_name]) for tag_name in cols
    }
    rc = RewardConstructor(reward_component_constructors)

    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )
    # call reward constructor
    pf = rc(pf)

    expected_cols = pd.Index(["tag-1", "tag-2", "reward"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 0,      np.nan],
            [0,      2,      0],
            [1,      4,      0],
            [2,      6,      2],
            [np.nan, np.nan, np.nan],
            [4,      10,     4],
            [5,      12,     5],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(pf.data, expected_df)


def test_null_filter():
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xform.GreaterThanConfig(threshold=5),
            ],
        ),
    ]

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)

    assert "tag-1" not in reward_constructor.component_constructors
    assert "tag-2" in reward_constructor.component_constructors


@dataclass
class EpcorRewardConfig:
    e_min: float = 85
    e_target: float = 95
    c_min: float = 0
    orp_pumpspeed_max: float = 60
    ph_pumpspeed_max: float = 60
    orp_cost_factor: float = 1.355 # $/(rpm*hr)
    ph_cost_factor: float = 0.5455 # $/(rpm*hr)
    r_e_min: float = 0 # r_e(e_min)
    r_e_target: float = 0.5 # r_e(e_max)
    r_c_min: float = 1.0 # r_c(c_min)
    r_c_max: float = 0.5 # r_c(c_max)
    high_pumpspeed_penalty: float = -1

def get_max_cost(cfg: EpcorRewardConfig) -> float:
    orp_cost_max = cfg.orp_pumpspeed_max * cfg.orp_cost_factor # $/hr
    ph_cost_max = cfg.ph_pumpspeed_max * cfg.ph_cost_factor # $/hr
    c_max = orp_cost_max + ph_cost_max # $/hr

    return c_max


def epcor_scrubber_reward(
    efficiency: float, orp_pumpspeed: float, ph_pumpspeed: float, cfg: EpcorRewardConfig
) -> float:
    m_e = (cfg.r_e_target - cfg.r_e_min) / (cfg.e_target - cfg.e_min)
    b_e = cfg.r_e_min - m_e * cfg.e_min

    def r_e(efficiency: float) -> float:
        return m_e * efficiency + b_e

    c_max = get_max_cost(cfg)
    m_c = (cfg.r_c_max - cfg.r_c_min) / (c_max - cfg.c_min)
    b_c = cfg.r_c_min - m_c * cfg.c_min

    def r_c(cost: float) -> float:
        return m_c * cost + b_c

    def r() -> float:
        penalty = 0
        if orp_pumpspeed > cfg.orp_pumpspeed_max:
            penalty += cfg.high_pumpspeed_penalty
        if ph_pumpspeed > cfg.ph_pumpspeed_max:
            penalty += cfg.high_pumpspeed_penalty
        if efficiency < cfg.e_target:
            return r_e(efficiency) + penalty
        else:
            cost = orp_pumpspeed * cfg.orp_cost_factor + ph_pumpspeed * cfg.ph_cost_factor
            return r_c(cost) + penalty

    return r()


def test_epcor_reward():

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(8)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["efficiency", "orp_pumpspeed", "ph_pumpspeed"])
    df = pd.DataFrame(
        data=[
            [np.nan, np.nan,   16],
            [85,      10,      10],
            [90,      20,      15],
            [95,      30,      25],
            [np.nan, np.nan, np.nan],
            [97,      45,      50],
            [98,      60,      65],
            [99,      70,      80],
        ],
        columns=cols,
        index=idx,
    )

    r_cfg = EpcorRewardConfig()
    m_e = (r_cfg.r_e_target - r_cfg.r_e_min) / (r_cfg.e_target - r_cfg.e_min)
    b_e = r_cfg.r_e_min - m_e * r_cfg.e_min

    c_max = get_max_cost(r_cfg)
    m_c = (r_cfg.r_c_max - r_cfg.r_c_min) / (c_max - r_cfg.c_min)
    b_c = r_cfg.r_c_min - m_c * r_cfg.c_min

    transform_cfgs = {
        "efficiency": [
            xform.AffineConfig(scale=m_e, bias=b_e),
            xform.ProductConfig(
                other="efficiency",
                other_xform=[
                    xform.LessThanConfig(
                        threshold=r_cfg.e_target,
                    ),
                ],
            ),
        ],
        "ph_pumpspeed": [
            xform.SplitConfig(
                passthrough=False,
                # passthrough=True,
                # left is high pumpspeed penalty
                left=[
                    xform.GreaterThanConfig(
                        threshold=r_cfg.ph_pumpspeed_max,
                    ),
                    xform.ScaleConfig(
                        factor=r_cfg.high_pumpspeed_penalty,
                    ),
                ],
                # right is ph component of cost reward
                right=[
                    xform.ScaleConfig(
                        factor=r_cfg.ph_cost_factor
                    ),
                    xform.AffineConfig(
                        scale=m_c,
                        bias=b_c/2 # add half of b to each pump
                    ),
                    xform.ProductConfig(
                        other="efficiency",
                        other_xform=[
                            xform.GreaterThanConfig(
                                threshold=r_cfg.e_target,
                                equal=True,
                            ),
                        ],
                    ),
                ],
            ), # end ph pumpspeed split config
        ],
        "orp_pumpspeed": [
            xform.SplitConfig(
                passthrough=False,
                # passthrough=True,
                # left is high pumpspeed penalty
                left=[
                    xform.GreaterThanConfig(
                        threshold=r_cfg.orp_pumpspeed_max,
                    ),
                    xform.ScaleConfig(
                        factor=r_cfg.high_pumpspeed_penalty,
                    ),
                ],
                # right is orp component of cost reward
                right=[
                    xform.ScaleConfig(
                        factor=r_cfg.orp_cost_factor
                    ),
                    xform.AffineConfig(
                        scale=m_c,
                        bias=b_c/2 # add half of b to each pump
                    ),
                    xform.ProductConfig(
                        other="efficiency",
                        other_xform=[
                            xform.GreaterThanConfig(
                                threshold=r_cfg.e_target,
                                equal=True,
                            ),
                        ],
                    ),
                ],
            ), # end ph pumpspeed split config
        ],
    }
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs[tag_name]) for tag_name in cols
    }
    rc = RewardConstructor(reward_component_constructors)

    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )
    # call reward constructor
    pf = rc(pf)

    r = epcor_scrubber_reward
    expected_rewards = [
        r(**_sanitize_dict(row), cfg=r_cfg)
        for row in df.to_dict(orient="records")
    ]
    expected_reward_df = pd.DataFrame(
        data=expected_rewards,
        columns=pd.Index(["reward"]),
        index=idx
    )
    expected_df = pd.concat([df, expected_reward_df], axis=1)

    assert dfs_close(pf.data, expected_df)


def _sanitize_dict[T](d: dict[Hashable, T]) -> dict[str, T]:
    """
    Because dict is invariant, passing a dict[A, ...] to a method expecting
    dict[B, ...] is invalid -- regardless of whether A is a subset or superset of B.
    This method converts `Hashable`s (typically coming from pandas) into `str`s
    in order to satisfy methods expecting a `dict[str, T]`
    """
    return {
        str(k): v
        for k, v in d.items()
    }
