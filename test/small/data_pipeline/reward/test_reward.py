import datetime
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pytest

import corerl.data_pipeline.transforms as xform
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.constructors.rc import RewardConstructor
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from test.infrastructure.utils.pandas import dfs_close


@pytest.fixture
def tag_cfgs():
    return [
        TagConfig(
            name='obs-1',
            preprocess=[],
        ),
        TagConfig(
            name='obs-2',
            preprocess=[],
        )
    ]


@pytest.fixture
def prep_stage(tag_cfgs: list[TagConfig]):
    return Preprocessor(tag_cfgs)


def test_rc1(tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
    raw_obs = pd.DataFrame(
        {
            "obs-1": [np.nan, 1, 2, 3, np.nan, 1, 2],
            "obs-2": [1, 2, 3, np.nan, 1, 2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    for cfg in tag_cfgs:
        cfg.reward_constructor = [
            xform.NormalizerConfig(min=0.0, max=1.0, from_data=False),
            xform.TraceConfig(trace_values=[0.1]),
        ]

    rc = RewardConstructor(tag_cfgs, prep_stage)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs-1_trace-0.1": [np.nan, 1.0, 1.9, 2.89, np.nan, 1.0, 1.9],
            "obs-2_trace-0.1": [1.0, 1.9, 2.89, np.nan, 1.0, 1.9, np.nan],
        }
    )
    expected_reward_vals = expected_components.sum(axis=1, skipna=False)
    expected_reward_df = pd.DataFrame({"reward": expected_reward_vals})

    assert dfs_close(pf.rewards, expected_reward_df)


def test_null_xform(tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
    raw_obs = pd.DataFrame(
        {
            "obs-1": [np.nan, 1, 2, 3, np.nan, 1, 2],
            "obs-2": [1, 2, 3, np.nan, 1, 2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    tag_cfgs[0].reward_constructor = [
        xform.NormalizerConfig(min=0.0, max=1.0, from_data=False),
        xform.TraceConfig(trace_values=[0.1]),
    ]
    tag_cfgs[1].reward_constructor = [
        xform.NullConfig(),
    ]

    # change final xform to null
    rc = RewardConstructor(tag_cfgs, prep_stage)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs_1_trace-0.1": [np.nan, 1.0, 1.9, 2.89, np.nan, 1.0, 1.9],
        }
    )
    expected_reward_vals = expected_components.sum(axis=1, skipna=False)
    expected_reward_df = pd.DataFrame({"reward": expected_reward_vals})

    assert dfs_close(pf.rewards, expected_reward_df)


def test_lessthan_xform():
    tag_cfgs = [
        TagConfig(
            name="obs-1",
            preprocess=[],
            # default null reward config
        ),
        TagConfig(
            name="obs-2",
            preprocess=[],
            reward_constructor=[
                xform.LessThanConfig(threshold=5),
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["obs-1", "obs-2"])
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

    prep_stage = Preprocessor(tag_cfgs)
    reward_constructor = RewardConstructor(tag_cfgs, prep_stage)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([True, True, True, False, np.nan, False, False])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)

    assert dfs_close(pf.rewards, expected_reward_df)


def test_greaterthan_xform():
    tag_cfgs = [
        TagConfig(
            name="obs-1",
            preprocess=[],
            # default null reward config
        ),
        TagConfig(
            name="obs-2",
            preprocess=[],
            reward_constructor=[
                xform.GreaterThanConfig(threshold=5),
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["obs-1", "obs-2"])
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

    prep_stage = Preprocessor(tag_cfgs)
    reward_constructor = RewardConstructor(tag_cfgs, prep_stage)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([False, False, False, True, np.nan, True, True])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)

    assert dfs_close(pf.rewards, expected_reward_df)


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
            name="obs-1",
            preprocess=[],
            # default null reward config
        ),
        TagConfig(
            name="obs-2",
            preprocess=[],
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

    cols = pd.Index(["obs-1", "obs-2"])
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

    prep_stage = Preprocessor(tag_cfgs)
    reward_constructor = RewardConstructor(tag_cfgs, prep_stage)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([0.0, 0.0, 0.0, -10.0, np.nan, -10.0, -10.0])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)

    assert dfs_close(pf.rewards, expected_reward_df)


def test_product_transform(tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
    xform.register_dispatchers()

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["obs-1", "obs-2"])
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

    tag_cfgs[0].reward_constructor = [
        xform.BinaryConfig(op="prod", other="obs-2", other_xform=[xform.GreaterThanConfig(threshold=5)])
    ]
    tag_cfgs[1].reward_constructor = [xform.NullConfig()]

    rc = RewardConstructor(tag_cfgs, prep_stage)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )
    # call reward constructor
    pf = rc(pf)

    expected_cols = pd.Index(["reward"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0],
            [0],
            [2],
            [np.nan],
            [4],
            [5],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(pf.rewards, expected_df)


@dataclass
class EpcorRewardConfig:
    outlet_h2s_target: float = 1
    max_outlet_h2s: float = 5
    e_min: float = 80
    e_target: float = 90
    c_min: float = 0
    orp_pumpspeed_max: float = 60
    ph_pumpspeed_max: float = 60
    orp_cost_factor: float = 1.355 # $/(rpm*hr)
    ph_cost_factor: float = 0.5455 # $/(rpm*hr)
    high_pumpspeed_penalty: float = -1

def get_max_cost(cfg: EpcorRewardConfig) -> float:
    orp_cost_max = cfg.orp_pumpspeed_max * cfg.orp_cost_factor # $/hr
    ph_cost_max = cfg.ph_pumpspeed_max * cfg.ph_cost_factor # $/hr
    c_max = orp_cost_max + ph_cost_max # $/hr

    return c_max


def get_constraint_violation_loss(efficiency: float, outlet_h2s: float, cfg: EpcorRewardConfig) -> float:
    """
    constraint: efficiency > e_target OR outlet_h2s < outlet_h2s_target
    """
    # transform to minimization
    # x > t  -->  -x < -t = z < g
    z1 = -efficiency
    g1 = -cfg.e_target

    # outlet h2s is already a minimization (outlet_)
    z2 = outlet_h2s
    g2 = cfg.outlet_h2s_target

    zs = [z1, z2]
    gs = [g1, g2]

    # get normalized constraint violation
    z1_max = -cfg.e_min
    z2_max = cfg.max_outlet_h2s
    z_maxs = [z1_max, z2_max]

    v_hats = []
    for z, g, z_max in zip(zs, gs, z_maxs, strict=True):
        v_hat = (z - g) / (z_max - g) # or affine 1/(z_max-g)*z - g/(z_max -g)
        v_hats.append(v_hat)

    # loss aggregates normalized constraint violations
    # for epcor scrubber, we only need one of the constraints to be satisfied
    # so we aggregate with a min
    L_v = min(v_hats) # \in [0, 1]
    assert isinstance(L_v, float)

    return L_v

def get_constraint_violation_reward(L_v: float) -> float:
    # transform to maximization
    r_v_raw = -L_v # \in [-1, 0]

    # normalize reward
    r_v_norm = r_v_raw + 1 # \in [0, 1]

    # compress to [-1, -0.5]
    r_v_prime = 0.5 * r_v_norm - 1

    # ignore if constraints are satisfied:
    # if aggregated violation L_v <= 0, this will evaluate to 0
    r_v = (L_v > 0) * r_v_prime

    return r_v


def get_optimization_reward(
    x: float, x_min: float, x_max: float, direction: Literal["min", "max"], L_v: float
) -> float:
    """
    While satisfying constraints, minimize chem cost
    """
    # transform to maximization
    if direction == 'min':
        y = -x
        y_max = -x_min
        y_min = -x_max
    else:
        y = x
        y_max = x_max
        y_min = x_min

    # normalize
    r_o_norm = (y - y_min) / (y_max - y_min)

    # compress to [-0.5, 0]
    r_o_prime = 0.5 * r_o_norm - 0.5

    # ignore if constraints are not satisfied:
    # if aggregated violation L_v > 0, this will evaluate to 0
    r_o = (L_v <= 0) * r_o_prime

    return r_o

def epcor_scrubber_reward(
    efficiency: float, outlet_h2s: float, orp_pumpspeed: float, ph_pumpspeed: float, cfg: EpcorRewardConfig
) -> float:

    cost = orp_pumpspeed * cfg.orp_cost_factor + ph_pumpspeed * cfg.ph_cost_factor
    c_max = get_max_cost(cfg)
    c_min = cfg.c_min

    constraint_violation_loss = get_constraint_violation_loss(efficiency, outlet_h2s, cfg)
    r_v = get_constraint_violation_reward(constraint_violation_loss)
    r_o = get_optimization_reward(x=cost, x_min=c_min, x_max=c_max, direction='min', L_v=constraint_violation_loss)

    r_base = r_o + r_v

    penalty = 0
    if orp_pumpspeed > cfg.orp_pumpspeed_max:
        penalty += cfg.high_pumpspeed_penalty
    if ph_pumpspeed > cfg.ph_pumpspeed_max:
        penalty += cfg.high_pumpspeed_penalty

    r = r_base + penalty

    return r


def test_epcor_reward():
    xform.register_dispatchers()

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(10)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["efficiency", "outlet_h2s", "orp_pumpspeed", "ph_pumpspeed"])
    df = pd.DataFrame(
        data=[
            [np.nan, 2,      np.nan,     16],
            [80,     3,       10,        10],
            [82,     1,       20,        15],
            [84,     0.98,    30,        25],
            [np.nan, np.nan, np.nan, np.nan],
            [89,     0.9,     45,        50],
            [88,     1.2,     60,        65],
            [99,     0.8,     70,        80],
            [88,     1.2,     40,        40],
            [99,     0.8,     40,        40],
        ],
        columns=cols,
        index=idx,
    )

    r_cfg = EpcorRewardConfig()

    c_min = 0
    c_max = get_max_cost(r_cfg)

    # used in derivation of normalized
    # constraint violation
    z1_max = -r_cfg.e_min
    g1 = -r_cfg.e_target

    z2_max = r_cfg.max_outlet_h2s
    g2 = r_cfg.outlet_h2s_target

    # xforms for normalized constraint violation from efficiency
    # re-used a few times
    constraint_violation_xforms = [
        xform.ScaleConfig(factor=-1), # transform to minimization (z)
        # next step gets normalized constraint violation
        xform.AffineConfig(
            scale=1/(z1_max - g1),
            bias=-g1/(z1_max - g1)
        ), # this gives v_hat1 \in [0, 1]
        xform.BinaryConfig(
            op="min",
            other="outlet_h2s",
            other_xform=[
                # next step gets normalized constraint violation
                xform.AffineConfig(
                    scale=1/(z2_max - g2),
                    bias=-g2/(z2_max - g2)
                ), # this gives v_hat1 \in [0, 1]
            ] # this gives v_hat2 \in [0, 1]
        )
    ] # this whole chain gives constrain violation L_v \in [0, 1]

    transform_cfgs: dict[str, list[xform.TransformConfig]] = {
        "efficiency": [
            *constraint_violation_xforms,
            xform.ScaleConfig(factor=-1), # maximization \in [-1, 0]
            xform.AffineConfig(bias=1), # normalize: max \in [0, 1]
            xform.AffineConfig(scale=0.5, bias=-1), # squash to [-1, -0.5]
            xform.BinaryConfig(
                op="prod",
                other="efficiency",
                other_xform=[
                    *constraint_violation_xforms,
                    xform.GreaterThanConfig(threshold=0),
                ],
            ),
        ],
        "outlet_h2s": [xform.NullConfig()], # this is hangled in constraint_violation_xforms above
        "ph_pumpspeed": [
            xform.SplitConfig(
                passthrough=False,
                # passthrough=True,
                # left is high pumpspeed penalty
                left=[
                    xform.GreaterThanConfig(threshold=r_cfg.ph_pumpspeed_max),
                    xform.ScaleConfig(factor=r_cfg.high_pumpspeed_penalty),
                ],
                # right is ph component of cost reward
                right=[
                    xform.ScaleConfig(factor=r_cfg.ph_cost_factor),
                    xform.BinaryConfig(
                        op="add",
                        other="orp_pumpspeed",
                        other_xform=[
                            xform.ScaleConfig(factor=r_cfg.orp_cost_factor),
                        ],
                    ),
                    # after the above add we have cost
                    xform.ScaleConfig(factor=-1), # transform to maximization
                    xform.AffineConfig( # normalize to [0, 1] maximization
                        scale=1/(c_max - c_min),
                        bias=c_max/(c_max - c_min ) # high/low flipped due to maximizaiton: y_min = -c_max
                    ),
                    xform.AffineConfig(scale=0.5, bias=-0.5), # squash to [-0.5, 0]
                    # next transform excludes this minimization reward if
                    # constraints are violated
                    xform.BinaryConfig(
                        op="prod",
                        other="efficiency",
                        other_xform=[
                            *constraint_violation_xforms,
                            xform.LessThanConfig(threshold=0, equal=True),
                        ],
                    ),
                ], # end of right
            ), # end ph pumpspeed split config
        ],
        # orp pumpspeed only has to handle high speed penalty
        "orp_pumpspeed": [
            xform.GreaterThanConfig(threshold=r_cfg.orp_pumpspeed_max),
            xform.ScaleConfig(factor=r_cfg.high_pumpspeed_penalty),
        ],
    }

    tag_cfgs = [
        TagConfig(
            name=name,
            preprocess=[],
            reward_constructor=xforms,
        )
        for name, xforms in transform_cfgs.items()
    ]
    prep_stage = Preprocessor(tag_cfgs)
    rc = RewardConstructor(tag_cfgs, prep_stage)

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

    assert dfs_close(pf.rewards, expected_reward_df)


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
