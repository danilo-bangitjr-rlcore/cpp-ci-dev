import datetime
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import corerl.data_pipeline.transforms as xform
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.constructors.rc import RewardConstructor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.tag_config import TagConfig
from corerl.state import AppState
from corerl.utils.maybe import Maybe


def dfs_close(df1: pd.DataFrame, df2: pd.DataFrame, col_order_matters: bool = False):
    if col_order_matters:
        if not df1.columns.equals(df2.columns):
            return False
    else:
        if set(df1.columns) != set(df2.columns):
            return False

    for col in df1.columns:
        if not np.allclose(df1[col], df2[col], equal_nan=True):
            return False

    return True

@dataclass
class EpcorRewardConfig:
    outlet_h2s_target: float = 1_000
    max_outlet_h2s: float = 5_000
    e_min: float = 85
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
        data_mode=DataMode.ONLINE,
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
    from e2e.make_configs import generate_tag_yaml
    generate_tag_yaml(Path("projects/epcor_scrubber"), tag_cfgs)

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
def test_epcor_reward_from_yaml(dummy_app_state: AppState):

    xform.register_dispatchers()

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(10)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["AI0879C", "AI0879B", "AIC3731_OUT", "AIC3730_OUT"])
    df: pd.DataFrame = pd.DataFrame(
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


    cfg = direct_load_config(
        MainConfig, base="projects/epcor_scrubber/configs/", config_name="epcor_scrubber_reward.yaml"
    )
    assert isinstance(cfg, MainConfig)

    pipeline = Pipeline(dummy_app_state, cfg.pipeline)
    pipe_return = pipeline(df, data_mode=DataMode.ONLINE)
    r_cfg = EpcorRewardConfig()
    r = epcor_scrubber_reward
    # cols = pd.Index(["AI0879C", "AI0879B", "AIC3731_OUT", "AIC3730_OUT"])
    nice_cols = {
        "AI0879C": "efficiency",
        "AI0879B": "outlet_h2s",
        "AIC3731_OUT": "orp_pumpspeed",
        "AIC3730_OUT": "ph_pumpspeed",
    }
    df = df.rename(columns=nice_cols)
    expected_rewards = [
        r(**_sanitize_dict(row), cfg=r_cfg)
        for row in df.to_dict(orient="records")
    ]
    expected_reward_df = pd.DataFrame(
        data=expected_rewards,
        columns=pd.Index(["reward"]),
        index=idx
    )
    print(pipe_return.rewards)
    print(expected_reward_df)
    assert dfs_close(pipe_return.rewards, expected_reward_df)

def get_bounds(tag_cfg: TagConfig):
    lo = (
        Maybe[float](tag_cfg.red_bounds and tag_cfg.red_bounds[0])
        .otherwise(lambda: tag_cfg.operating_range and tag_cfg.operating_range[0])
        .otherwise(lambda: tag_cfg.yellow_bounds and tag_cfg.yellow_bounds[0])
    ).expect()

    hi = (
        Maybe[float](tag_cfg.red_bounds and tag_cfg.red_bounds[1])
        .otherwise(lambda: tag_cfg.operating_range and tag_cfg.operating_range[1])
        .otherwise(lambda: tag_cfg.yellow_bounds and tag_cfg.yellow_bounds[1])
    ).expect()
    return lo, hi

def test_epcor_reward_rankings(dummy_app_state: AppState):
    """
    A vs B comparison of states
    """

    xform.register_dispatchers()

    cfg = direct_load_config(
        MainConfig, base="projects/epcor_scrubber/configs/", config_name="epcor_scrubber.yaml"
    )
    assert isinstance(cfg, MainConfig)

    pipeline = Pipeline(dummy_app_state, cfg.pipeline)
    tag_cfgs = cfg.pipeline.tags

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(1)]
    idx = pd.DatetimeIndex(dates)

    cols = [t.name for t in tag_cfgs]
    bounds = {}
    for tag_cfg in tag_cfgs:
        col = tag_cfg.name
        bounds[col] = get_bounds(tag_cfg)

    def red_zone_violated(df: pd.DataFrame):
        if df["AIC3731_OUT"].iloc[0] > 60:
            # pump speed penalty
            return True

        if df["AIC3730_OUT"].iloc[0] > 60:
            # pump speed penalty
            return True

        return False

    def tawws_violated(df: pd.DataFrame):
        "threshold at which we switch (TAWWS)"
        return df["AI0879C"].iloc[0] < 90 and df["AI0879B"].iloc[0] > 1000

    def single_checks(df: pd.DataFrame, reward: float):
        assert reward <= 0
        if red_zone_violated(df):
            assert reward <= -1

        if tawws_violated(df):
            # constraints violated
            assert reward < -0.5
        else:
            # constraints satisfied
            assert reward >= -0.5

    for _ in range(1000):
        A_vals = [[np.random.uniform(*bounds[col]) for col in cols]]
        A_df: pd.DataFrame = pd.DataFrame(
            data=A_vals,
            columns=cols,
            index=idx,
        )

        B_vals = [[np.random.uniform(*bounds[col]) for col in cols]]
        B_df: pd.DataFrame = pd.DataFrame(
            data=B_vals,
            columns=cols,
            index=idx,
        )

        # call reward constructor
        A_reward = pipeline(A_df, data_mode=DataMode.ONLINE).rewards.to_numpy()[0]
        B_reward = pipeline(B_df, data_mode=DataMode.ONLINE).rewards.to_numpy()[0]
        single_checks(A_df, A_reward)
        single_checks(B_df, B_reward)

        if red_zone_violated(A_df) and red_zone_violated(B_df):
            # not worth comparing
            continue

        if tawws_violated(A_df) and not tawws_violated(B_df):
            assert B_reward > A_reward

        if tawws_violated(B_df) and not tawws_violated(A_df):
            assert A_reward > B_reward

        if not tawws_violated(B_df) and not tawws_violated(A_df):
            A_cost = (
                EpcorRewardConfig.orp_cost_factor * A_df["AIC3731_OUT"].iloc[0]
                + EpcorRewardConfig.ph_cost_factor * A_df["AIC3730_OUT"].iloc[0]
            )
            B_cost = (
                EpcorRewardConfig.orp_cost_factor * B_df["AIC3731_OUT"].iloc[0]
                + EpcorRewardConfig.ph_cost_factor * B_df["AIC3730_OUT"].iloc[0]
            )
            # A is only greater reward if it has lower cost
            assert (A_reward > B_reward) == (A_cost < B_cost)
