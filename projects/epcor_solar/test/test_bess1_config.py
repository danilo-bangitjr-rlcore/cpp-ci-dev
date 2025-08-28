import datetime
import numpy as np
import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.deltaize_tags import DeltaizeTags
from corerl.data_pipeline.seasonal_tags import SeasonalTagIncluder
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.tags.components.bounds import get_widest_static_bounds
from corerl.tags.tag_config import get_scada_tags
from corerl.tags.validate_tag_configs import validate_tag_configs
from corerl.data_pipeline.virtual_tags import VirtualTagComputer
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState

from libs.lib_config.lib_config.loader import direct_load_config
from libs.lib_defs.lib_defs.config_defs.tag_config import TagType

@pytest.fixture
def cfg():
    main_cfg = direct_load_config(
        MainConfig,
        config_name='epcor_solar/configs/bess1.yaml',
    )
    validate_tag_configs(main_cfg)
    main_cfg.metrics.enabled = False
    main_cfg.evals.enabled = False
    # Want to avoid pipeline warmup in tests
    trace_cfg = TraceConfig(
        trace_values=[0.0, 0.75, 0.95, 0.99],
        missing_tol=1.0
    )
    main_cfg.pipeline.state_constructor.defaults = [trace_cfg]

    return main_cfg

@pytest.fixture
def app_state(cfg: MainConfig):
    return AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=DummyEventBus(),
    )

@pytest.fixture
def tag_names(cfg: MainConfig):
    opc_tags = get_scada_tags(cfg.pipeline.tags)
    return [tag.name for tag in opc_tags]

@pytest.fixture
def df(cfg: MainConfig, tag_names: list[str]):
    # Assuming obs_period = 5 minutes and action_period = 5 minutes
    obs_period = cfg.interaction.obs_period
    obs_steps = 3
    start = datetime.datetime(2025, 1, 22, 10)
    dates = [start + i * obs_period for i in range(obs_steps)]

    data = pd.DataFrame(data=np.ones((len(dates), len(tag_names))),
                      columns=tag_names,
                      index=dates)

    for tag_cfg in cfg.pipeline.tags:
        if tag_cfg.name in tag_names:
            for i in range(obs_steps):
                tag_bounds = get_widest_static_bounds(tag_cfg)
                if tag_cfg.type == TagType.delta:
                    data.loc[dates[i], tag_cfg.name] += i * tag_bounds[1].unwrap()
                else:
                    data.loc[dates[i], tag_cfg.name] *= tag_bounds[1].unwrap()

    # Costs
    data["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    data["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    data["Actual_Posted_Pool_Price"] = 111.11
    data["ELS_SLR_ENG_EXPORT_FEE_SP"] = 0.375

    # Red/Yellow Zone Tags
    data["ELS_SLR_ELSP72_POWER_PEAK_SP_LGC"] = 4.4
    data["ELS_SLR_ELSP72_KW"] = 4000
    data["ELS_SLR_ELSP72_MAJORITY_LGC"] = 100
    data["ELS_SLR_ELSP72_ENG_IN_TOT"] *= 0.0
    for step in range(obs_steps):
        data.loc[dates[step], "ELS_SLR_ELSP72_ENG_IN_TOT"] += 0.2 * step

    data["ELS_SLR_BESS1_ENG_OUT_TOT"] = 0

    return data

def test_P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_early_in_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.ones((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["2/14/2025 11:00"]))
    df["ELS_SLR_ELSP72_MAJORITY_LGC"] = 85.7
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 45
    assert pf.data["day_of_week"].iloc[0] == 4
    assert pf.data["time_of_day"].iloc[0] == 39600

    out = virtual_tag_stage(pf)

    # '(100 - {ELS_SLR_ELSP72_MAJORITY_LGC}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {ELS_SLR_ELSP72_MAJORITY_LGC} = 85.7
    # {day_of_year} = 45
    # (1 / (1 + 2.71828**(-1*((45 - 250) / 20)))) = 0.000035
    # (100 - 85.7) * 0.000035 = 0.0005005
    assert np.isclose(out.data["P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT"], 0.0005005, atol=1e-4)

def test_P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_mid_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.ones((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["6/24/2025 19:27"]))
    df["ELS_SLR_ELSP72_MAJORITY_LGC"] = 20.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 175
    assert pf.data["day_of_week"].iloc[0] == 1
    assert pf.data["time_of_day"].iloc[0] == 70020

    out = virtual_tag_stage(pf)

    # '(100 - {ELS_SLR_ELSP72_MAJORITY_LGC}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {ELS_SLR_ELSP72_MAJORITY_LGC} = 20
    # {day_of_year} = 175
    # (1 / (1 + 2.71828**(-1*((175 - 250) / 20)))) = 0.022977
    # (100 - 20) * 0.022977 = 1.83816
    assert np.isclose(out.data["P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT"], 1.83816, atol=1e-4)

def test_P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_end_of_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.ones((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["12/11/2025 6:00"]))
    df["ELS_SLR_ELSP72_MAJORITY_LGC"] = 50.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 345
    assert pf.data["day_of_week"].iloc[0] == 3
    assert pf.data["time_of_day"].iloc[0] == 21600

    out = virtual_tag_stage(pf)

    # '(100 - {ELS_SLR_ELSP72_MAJORITY_LGC}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {ELS_SLR_ELSP72_MAJORITY_LGC} = 50
    # {day_of_year} = 345
    # (1 / (1 + 2.71828**(-1*((345 - 250) / 20)))) = 0.991422
    # (100 - 50) * 0.991422 = 49.5711
    assert np.isclose(out.data["P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT"], 49.5711, atol=1e-4)

def test_BESS_DEGRADATION_COST(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05"])
    df = pd.DataFrame(data=np.ones((2, len(tag_names))), columns=tag_names, index=dates)
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df.loc[dates[0], "ELS_SLR_BESS1_ENG_IN_TOT"] = 100.0
    df.loc[dates[1], "ELS_SLR_BESS1_ENG_IN_TOT"] = 105.0
    df.loc[dates[0], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0
    df.loc[dates[1], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)
    assert np.isnan(pf.data["ELS_SLR_BESS1_ENG_IN_TOT"].iloc[0])
    assert np.isclose(pf.data["ELS_SLR_BESS1_ENG_IN_TOT"].iloc[1], 5.0)
    assert np.isnan(pf.data["ELS_SLR_BESS1_ENG_OUT_TOT"].iloc[0])
    assert np.isclose(pf.data["ELS_SLR_BESS1_ENG_OUT_TOT"].iloc[1], 0.0)

    out = virtual_tag_stage(pf)

    # '{ELS_SLR_ENG_BATT_COST_SP} * ({ELS_SLR_BESS1_ENG_IN_TOT} + {ELS_SLR_BESS1_ENG_OUT_TOT})'
    # 5 * (5 + 0) = 25
    assert np.isnan(out.data["BESS_DEGRADATION_COST"].iloc[0])
    assert np.isclose(out.data["BESS_DEGRADATION_COST"].iloc[1], 25.0)

def test_CHARGING_FROM_GRID(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_IN_TOT"] = 100.0 + (i * 0.1)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # CHARGING_FROM_GRID: '({ELS_SLR_SOLAR1_ENG_OUT_TOT} <= {ELS_SLR_WTP1_ENG_IN_TOT}) & ({ELS_SLR_BESS1_ENG_IN_TOT} > 0)'
    # (0.5 <= 0.3) & (0.1 > 0): False
    # (0.5 <= 0.9) & (0.1 > 0): True
    assert not out.data["CHARGING_FROM_GRID"].iloc[1]
    assert out.data["CHARGING_FROM_GRID"].iloc[2]

def test_CHARGING_FROM_SOLAR(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_IN_TOT"] = 100.0 + (i * 0.1)
    df["ELS_SLR_ELSP72_KW"] = 4800
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # CHARGING_FROM_SOLAR: '(({ELS_SLR_SOLAR1_ENG_OUT_TOT} > {ELS_SLR_WTP1_ENG_IN_TOT}) & ({ELS_SLR_BESS1_ENG_IN_TOT} > 0)) & ({ELS_SLR_ELSP72_KW} < 4875)'
    # (0.5 > 0.3) & (0.1 > 0) & (4800 < 4875): True
    # (0.5 > 0.9) & (0.1 > 0) & (4800 < 4875): False
    assert out.data["CHARGING_FROM_SOLAR"].iloc[1]
    assert not out.data["CHARGING_FROM_SOLAR"].iloc[2]

def test_CHARGING_FROM_GRID_COST(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    df["Actual_Posted_Pool_Price"] = 100.0
    df["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df["ELS_SLR_BESS1_ENG_OUT_TOT"] = 0.0
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_IN_TOT"] = 100.0 + (i * 0.1)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # CHARGING_FROM_GRID_COST: '{CHARGING_FROM_GRID} * {ELS_SLR_BESS1_ENG_IN_TOT} * {ELS_SLR_ENG_IMPORT_COST_SP}'
    # False * 0.1 * 47.42 = 0.0
    # True * 0.1 * 47.42 = 4.742
    # BESS_DEGRADATION: 0.1 * 5.0 = 0.5
    # i=1 CHARGING_FROM_SOLAR_COST = 0.1 * 100.0 = 10.0
    # i=2 CHARGING_FROM_SOLAR_COST = 0.0
    assert np.isclose(out.data["CHARGING_FROM_GRID_COST"].iloc[1], 0.0)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[1], -10.0)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[1], -10.5)
    assert np.isclose(out.data["CHARGING_FROM_GRID_COST"].iloc[2], 4.742)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[2], -4.742)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[2], -5.242)

def test_CHARGING_FROM_SOLAR_COST(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    df["Actual_Posted_Pool_Price"] = 100.0
    df["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df["ELS_SLR_BESS1_ENG_OUT_TOT"] = 0.0
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_IN_TOT"] = 100.0 + (i * 0.1)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # CHARGING_FROM_SOLAR_COST: '{CHARGING_FROM_SOLAR} * {ELS_SLR_BESS1_ENG_IN_TOT} * {Actual_Posted_Pool_Price}'
    # True * 0.1 * 100.0 = 10.0
    # False * 0.1 * 100.0 = 0.0
    # BESS_DEGRADATION: 0.1 * 5.0 = 0.5
    # i=1 CHARGING_FROM_GRID_COST = 0.0
    # i=2 CHARGING_FROM_GRID_COST = 0.1 * 47.42 = 4.742
    assert np.isclose(out.data["CHARGING_FROM_SOLAR_COST"].iloc[1], 10.0)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[1], -10.0)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[1], -10.5)
    assert np.isclose(out.data["CHARGING_FROM_SOLAR_COST"].iloc[2], 0.0)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[2], -4.742)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[2], -5.242)

def test_ALL_ENERGY_DISCHARGED_POOL_PRICE(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.1)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # ALL_ENERGY_DISCHARGED_POOL_PRICE: '({ELS_SLR_SOLAR1_ENG_OUT_TOT} > {ELS_SLR_WTP1_ENG_IN_TOT}) & ({ELS_SLR_BESS1_ENG_OUT_TOT} > 0)'
    # (0.5 > 0.3) & (0.1 > 0): True
    # (0.5 > 0.9) & (0.1 > 0): False
    assert out.data["ALL_ENERGY_DISCHARGED_POOL_PRICE"].iloc[1]
    assert not out.data["ALL_ENERGY_DISCHARGED_POOL_PRICE"].iloc[2]

def test_ALL_ENERGY_DISCHARGED_GRID_PRICE(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.05)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # ALL_ENERGY_DISCHARGED_GRID_PRICE: '(({ELS_SLR_SOLAR1_ENG_OUT_TOT} + {ELS_SLR_BESS1_ENG_OUT_TOT}) <= {ELS_SLR_WTP1_ENG_IN_TOT}) & ({ELS_SLR_BESS1_ENG_OUT_TOT} > 0)'
    # (0.5 + 0.05 <= 0.3) & (0.05 > 0): False
    # (0.5 + 0.05 <= 0.9) & (0.05 > 0): True
    assert not out.data["ALL_ENERGY_DISCHARGED_GRID_PRICE"].iloc[1]
    assert out.data["ALL_ENERGY_DISCHARGED_GRID_PRICE"].iloc[2]

def test_SOME_ENERGY_DISCHARGED_POOL_PRICE(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.475)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.05)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.5)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # SOME_ENERGY_DISCHARGED_POOL_PRICE: '((({ELS_SLR_SOLAR1_ENG_OUT_TOT} + {ELS_SLR_BESS1_ENG_OUT_TOT}) > {ELS_SLR_WTP1_ENG_IN_TOT}) & ({ELS_SLR_SOLAR1_ENG_OUT_TOT} < {ELS_SLR_WTP1_ENG_IN_TOT})) & ({ELS_SLR_BESS1_ENG_OUT_TOT} > 0)'
    # (0.475 + 0.05 > 0.5) & (0.475 < 0.5) & (0.05 > 0): True
    # (0.475 + 0.05 > 1.5) & (0.475 < 1.5) & (0.05 > 0): False
    assert out.data["SOME_ENERGY_DISCHARGED_POOL_PRICE"].iloc[1]
    assert not out.data["SOME_ENERGY_DISCHARGED_POOL_PRICE"].iloc[2]

def test_DISCHARGED_POOL_PRICE_REVENUE_1(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    df["Actual_Posted_Pool_Price"] = 100.0
    df["ELS_SLR_ENG_EXPORT_FEE_SP"] = 0.375
    df["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df["ELS_SLR_BESS1_ENG_IN_TOT"] = 0.0
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.1)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i**2 * 0.2)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # DISCHARGED_POOL_PRICE_REVENUE: '({ALL_ENERGY_DISCHARGED_POOL_PRICE} * {ELS_SLR_BESS1_ENG_OUT_TOT} * ({Actual_Posted_Pool_Price} - {ELS_SLR_ENG_EXPORT_FEE_SP})) + ({SOME_ENERGY_DISCHARGED_POOL_PRICE} * ({ELS_SLR_SOLAR1_ENG_OUT_TOT} + {ELS_SLR_BESS1_ENG_OUT_TOT} - {ELS_SLR_WTP1_ENG_IN_TOT}) * ({Actual_Posted_Pool_Price} - {ELS_SLR_ENG_EXPORT_FEE_SP}))'
    # (True * 0.1 * (100.0 - 0.375)) + (False * (0.5 + 0.1 - 0.2 * (100 - 0.375)) = 9.9625
    # (False * 0.1 * (100.0 - 0.375)) + (False * (0.5 + 0.1 - 0.6 * (100 - 0.375)) = 0
    # BESS_DEGRADATION: 0.1 * 5.0 = 0.5
    # i=1 DISCHARGED GRID PRICE REVENUE: 0.0
    # i=2 DISCHARGED GRID PRICE REVENUE: 0.1 * 47.42 = 4.742
    assert np.isclose(out.data["DISCHARGED_POOL_PRICE_REVENUE"].iloc[1], 9.9625)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[1], 9.9625)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[1], 9.4625)
    assert np.isclose(out.data["DISCHARGED_POOL_PRICE_REVENUE"].iloc[2], 0.0)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[2], 4.742)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[2], 4.242)

def test_DISCHARGED_POOL_PRICE_REVENUE_2(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    df["Actual_Posted_Pool_Price"] = 100.0
    df["ELS_SLR_ENG_EXPORT_FEE_SP"] = 0.375
    df["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df["ELS_SLR_BESS1_ENG_IN_TOT"] = 0.0
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.475)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.05)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i ** 2 * 0.5)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # DISCHARGED_POOL_PRICE_REVENUE: '({ALL_ENERGY_DISCHARGED_POOL_PRICE} * {ELS_SLR_BESS1_ENG_OUT_TOT} * ({Actual_Posted_Pool_Price} - {ELS_SLR_ENG_EXPORT_FEE_SP})) + ({SOME_ENERGY_DISCHARGED_POOL_PRICE} * ({ELS_SLR_SOLAR1_ENG_OUT_TOT} + {ELS_SLR_BESS1_ENG_OUT_TOT} - {ELS_SLR_WTP1_ENG_IN_TOT}) * ({Actual_Posted_Pool_Price} - {ELS_SLR_ENG_EXPORT_FEE_SP}))'
    # (False * 0.05 * (100.0 - 0.375)) + (True * (0.475 + 0.05 - 0.5 * (100 - 0.375)) = 2.490625
    # (False * 0.05 * (100.0 - 0.375)) + (False * (0.475 + 0.05 - 1.5 * (100 - 0.375)) = 0
    # BESS_DEGRADATION: 0.05 * 5.0 = 0.25
    # i=1 DISCHARGED GRID PRICE REVENUE: (0.5 - 0.475) * 47.42 = 1.1855
    # i=2 DISCHARGED GRID PRICE REVENUE: 0.05 * 47.42 = 2.371
    assert np.isclose(out.data["DISCHARGED_POOL_PRICE_REVENUE"].iloc[1], 2.490625)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[1], 3.676125)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[1], 3.426125)
    assert np.isclose(out.data["DISCHARGED_POOL_PRICE_REVENUE"].iloc[2], 0.0)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[2], 2.371)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[2], 2.121)

def test_DISCHARGED_GRID_PRICE_REVENUE_1(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    df["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df["ELS_SLR_BESS1_ENG_IN_TOT"] = 0.0
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.5)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.05)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i ** 2 * 0.2)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # DISCHARGED_GRID_PRICE_REVENUE: '({ALL_ENERGY_DISCHARGED_GRID_PRICE} * {ELS_SLR_BESS1_ENG_OUT_TOT} * {ELS_SLR_ENG_IMPORT_COST_SP}) + ({SOME_ENERGY_DISCHARGED_POOL_PRICE} * ({ELS_SLR_WTP1_ENG_IN_TOT} - {ELS_SLR_SOLAR1_ENG_OUT_TOT}) * {ELS_SLR_ENG_IMPORT_COST_SP})'
    # (False * 0.05 * 47.42) + (False * ((0.2 - 0.5) * 47.42) = 0
    # (True * 0.05 * 47.42) + (False * ((0.6 - 0.5) * 47.42) = 2.371
    # BESS_DEGRADATION: 0.05 * 5.0 = 0.25
    assert np.isclose(out.data["DISCHARGED_GRID_PRICE_REVENUE"].iloc[1], 0.0)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[1], 0.0)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[1], -0.25)
    assert np.isclose(out.data["DISCHARGED_GRID_PRICE_REVENUE"].iloc[2], 2.371)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[2], 2.371)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[2], 2.121)

def test_DISCHARGED_GRID_PRICE_REVENUE_2(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05", "12/11/2025 6:10"])
    df = pd.DataFrame(data=np.ones((3, len(tag_names))), columns=tag_names, index=dates)
    df["ELS_SLR_ENG_IMPORT_COST_SP"] = 47.42
    df["ELS_SLR_ENG_BATT_COST_SP"] = 5.0
    df["ELS_SLR_BESS1_ENG_IN_TOT"] = 0.0
    for i in range(len(dates)):
        df.loc[dates[i], "ELS_SLR_SOLAR1_ENG_OUT_TOT"] = 100.0 + (i * 0.475)
        df.loc[dates[i], "ELS_SLR_BESS1_ENG_OUT_TOT"] = 100.0 + (i * 0.05)
        df.loc[dates[i], "ELS_SLR_WTP1_ENG_IN_TOT"] = 100.0 + (i ** 2 * 0.5)
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # DISCHARGED_GRID_PRICE_REVENUE: '({ALL_ENERGY_DISCHARGED_GRID_PRICE} * {ELS_SLR_BESS1_ENG_OUT_TOT} * {ELS_SLR_ENG_IMPORT_COST_SP}) + ({SOME_ENERGY_DISCHARGED_POOL_PRICE} * ({ELS_SLR_WTP1_ENG_IN_TOT} - {ELS_SLR_SOLAR1_ENG_OUT_TOT}) * {ELS_SLR_ENG_IMPORT_COST_SP})'
    # (False * 0.05 * 47.42) + (True * ((0.5 - 0.475) * 47.42) = 1.1855
    # (True * 0.05 * 47.42) + (False * ((1.5 - 0.475) * 47.42) = 2.371
    # BESS_DEGRADATION: 0.05 * 5.0 = 0.25
    assert np.isclose(out.data["DISCHARGED_GRID_PRICE_REVENUE"].iloc[1], 1.1855)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[1], 1.1855)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[1], 0.9355)
    assert np.isclose(out.data["DISCHARGED_GRID_PRICE_REVENUE"].iloc[2], 2.371)
    assert np.isclose(out.data["BESS_PROFIT"].iloc[2], 2.371)
    assert np.isclose(out.data["BESS_DEGRADATION_PROFIT"].iloc[2], 2.121)

def test_no_violations(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    pipeline = Pipeline(app_state, cfg.pipeline)

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * 0.1667 = 0.8335
    # DISCHARGED_POOL_PRICE_REVENUE = 0.0
    # DISCHARGED_GRID_PRICE_REVENUE = 0.0
    # CHARGING_FROM_SOLAR_COST = 0.0
    # CHARGING_FROM_GRID_COST = 0.1667 * 47.42 = 7.904914
    # BESS_PROFIT = (0 + 0) - (0 + 7.904914) = -7.904914
    # BESS_DEGRADATION_PROFIT = -7.904914 - 0.8335 = -8.738414
    # Normalized PROFIT = 0.764687
    # Scaled to [-1, 0]: -0.235313
    # With Return Normalization: -0.00235313
    # n-step reward: -0.00235313
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == 2
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.00235313, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.00235313, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.235313, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.235313, atol=1e-4)

def test_curtailment_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    pipeline = Pipeline(app_state, cfg.pipeline)

    # Red/Yellow Zone Tags
    df["ELS_SLR_ELSP72_KW"] = 4750 # Curtailment yellow zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * 0.1667 = 0.8335
    # DISCHARGED_POOL_PRICE_REVENUE = 0.0
    # DISCHARGED_GRID_PRICE_REVENUE = 0.0
    # CHARGING_FROM_SOLAR_COST = 0.0
    # CHARGING_FROM_GRID_COST = 0.1667 * 47.42 = 7.904914
    # BESS_PROFIT = (0 + 0) - (0 + 7.904914) = -7.904914
    # BESS_DEGRADATION_PROFIT = -7.904914 - 0.8335 = -8.738414
    # Normalized PROFIT = 0.764687
    # Scaled to [-1, 0]: -0.235313
    # ELS_SLR_ELSP72_KW = 4750 represents 16.67% yellow zone violation
    # Penalty = -2 * (0.1666 ** 2) = -0.05551112
    # Unnormalized Reward: -0.235313 + -0.05551112 = -0.29082412
    # With Return Normalization: -0.0029082412
    # n-step reward: -0.0029082412
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == 2
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.0029082412, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.0029082412, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.29082412, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.29082412, atol=1e-4)

def test_power_peak_yellow_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    obs_steps = 3
    dates = df.index.tolist()

    pipeline = Pipeline(app_state, cfg.pipeline)

    # Red/Yellow Zone Tags
    # Red Bound: '{ELS_SLR_ELSP72_POWER_PEAK_SP_LGC} * 5 / 60' = 4.4 * 5 / 60 = 0.36667
    # Yellow Bound: 0.95 * 0.36667 = 0.3483
    df["ELS_SLR_ELSP72_ENG_IN_TOT"] = 0.0
    for step in range(obs_steps):
        df.loc[dates[step], "ELS_SLR_ELSP72_ENG_IN_TOT"] += 0.35 * step # yellow zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * 0.1667 = 0.8335
    # DISCHARGED_POOL_PRICE_REVENUE = 0.0
    # DISCHARGED_GRID_PRICE_REVENUE = 0.0
    # CHARGING_FROM_SOLAR_COST = 0.0
    # CHARGING_FROM_GRID_COST = 0.1667 * 47.42 = 7.904914
    # BESS_PROFIT = (0 + 0) - (0 + 7.904914) = -7.904914
    # BESS_DEGRADATION_PROFIT = -7.904914 - 0.8335 = -8.738414
    # Normalized PROFIT = 0.764687
    # Scaled to [-1, 0]: -0.235313
    # ELS_SLR_ELSP72_ENG_IN_TOT = 0.35 represents 9.0909% yellow zone violation
    # Penalty = -2 * (0.090909 ** 2) = -0.01652889
    # Unnormalized Reward: -0.235313 + -0.01652889 = -0.25184189
    # With Return Normalization: -0.0025184189
    # n-step reward: -0.0025184189
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == 2
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.0025184189, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.0025184189, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.25184189, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.25184189, atol=1e-4)

def test_power_peak_red_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    obs_steps = 3
    dates = df.index.tolist()

    pipeline = Pipeline(app_state, cfg.pipeline)

    # Red/Yellow Zone Tags
    # Red Bound: '{ELS_SLR_ELSP72_POWER_PEAK_SP_LGC} * 5 / 60' = 4.4 * 5 / 60 = 0.36667
    # Yellow Bound: 0.95 * 0.36667 = 0.3483
    df["ELS_SLR_ELSP72_ENG_IN_TOT"] = 0.0
    for step in range(obs_steps):
        df.loc[dates[step], "ELS_SLR_ELSP72_ENG_IN_TOT"] += 0.4 * step  # red zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # ELS_SLR_ELSP72_ENG_IN_TOT = 0.4 represents 5.2631% red zone violation
    # Unnormalized Reward = -4 - (4 * 0.052626) = -4.210504
    # With Return Normalization: -0.04210504
    # n-step reward: -0.04210504
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == 2
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.04210504, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.04210504, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -4.210504, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -4.210504, atol=1e-4)

def test_on_site_consumption_yellow_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    pipeline = Pipeline(app_state, cfg.pipeline)

    obs_period = cfg.interaction.obs_period
    obs_steps = 3
    start = datetime.datetime(2025, 12, 22, 10)
    dates = [start + i * obs_period for i in range(obs_steps)]
    df.index = dates

    # Red/Yellow Zone Tags
    df["ELS_SLR_ELSP72_MAJORITY_LGC"] = 50 # On-site consumption yellow zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * 0.1667 = 0.8335
    # DISCHARGED_POOL_PRICE_REVENUE = 0.0
    # DISCHARGED_GRID_PRICE_REVENUE = 0.0
    # CHARGING_FROM_SOLAR_COST = 0.0
    # CHARGING_FROM_GRID_COST = 0.1667 * 47.42 = 7.904914
    # BESS_PROFIT = (0 + 0) - (0 + 7.904914) = -7.904914
    # BESS_DEGRADATION_PROFIT = -7.904914 - 0.8335 = -8.738414
    # Normalized PROFIT = 0.764687
    # Scaled to [-1, 0]: -0.235313
    # P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT:
    # '(100 - {ELS_SLR_ELSP72_MAJORITY_LGC}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # (100 - 50) * (1 / (1 + 2.71828**(-1*((356 - 250) / 20)))) = 49.751659
    # Represents 97.31438% yellow zone violation
    # Penalty = -2 * (0.97314 ** 2) = -1.894
    # Unnormalized Reward: -0.235313 + -1.894 = -2.129313
    # With Return Normalization: -0.02129313
    # n-step reward: -0.02129313
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == 2
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.02129313, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.02129313, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -2.129313, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -2.129313, atol=1e-4)

def test_on_site_consumption_red_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    pipeline = Pipeline(app_state, cfg.pipeline)

    obs_period = cfg.interaction.obs_period
    obs_steps = 3
    start = datetime.datetime(2025, 12, 22, 10)
    dates = [start + i * obs_period for i in range(obs_steps)]
    df.index = dates

    # Red/Yellow Zone Tags
    df["ELS_SLR_ELSP72_MAJORITY_LGC"] = 48 # On-site consumption red zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT:
    # '(100 - {ELS_SLR_ELSP72_MAJORITY_LGC}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # (100 - 48) * (1 / (1 + 2.71828**(-1*((356 - 250) / 20)))) = 51.7417
    # Represents 3.867978% red zone violation
    # Unnormalized Reward: -4 - (4 * 0.0386797) = -4.1547188
    # With Return Normalization: -0.041547188
    # n-step reward: -0.041547188
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == 2
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.041547188, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.041547188, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -4.1547188, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -4.1547188, atol=1e-4)
