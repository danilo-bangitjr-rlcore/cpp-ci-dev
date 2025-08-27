import datetime
import numpy as np
import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.deltaize_tags import DeltaizeTags
from corerl.data_pipeline.seasonal_tags import SeasonalTagIncluder
from corerl.data_pipeline.pipeline import Pipeline
from corerl.tags.components.bounds import get_priority_violation_bounds
from corerl.tags.tag_config import get_scada_tags
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
        config_name='epcor_solar/configs/epcor_solar.yaml',
    )
    main_cfg.metrics.enabled = False
    main_cfg.evals.enabled = False

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
    action_period = cfg.interaction.action_period
    obs_steps = int(action_period / obs_period) + 1
    start = datetime.datetime(2025, 1, 22, 10)
    dates = [start + i * obs_period for i in range(obs_steps)]

    data = pd.DataFrame(data=np.ones((len(dates), len(tag_names))),
                      columns=tag_names,
                      index=dates)

    for tag_cfg in cfg.pipeline.tags:
        if tag_cfg.name in tag_names:
            for i in range(obs_steps):
                tag_bounds = get_priority_violation_bounds(tag_cfg, data.iloc[[i]])
                if tag_cfg.type == TagType.delta:
                    data.loc[dates[i], tag_cfg.name] += i * tag_bounds[1].unwrap()
                else:
                    data.loc[dates[i], tag_cfg.name] *= tag_bounds[1].unwrap()

    # Costs
    data["SCADA1.ELS_SLR_ENG_BATT_COST_SP.F_CV"] = 5.0
    data["SCADA1.ELS_SLR_ENG_IMPORT_COST_SP.F_CV"] = 47.42
    data["Actual_Posted_Pool_Price"] = 111.11
    data["SCADA1.ELS_SLR_ENG_EXPORT_FEE_SP.F_CV"] = 0.375

    # Red/Yellow Zone Tags
    data["SCADA1.ELS_SLR_ELSP12_POWER_PEAK_SP_LGC.F_CV"] = 4400
    data["SCADA1.ELS_SLR_ELSP12_KW.F_CV"] = 4000
    data["SCADA1.ELS_SLR_ELSP72_POWER_PEAK_SP_LGC.F_CV"] = 3500
    data["SCADA1.ELS_SLR_ELSP72_KW.F_CV"] = 4000
    data["SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV"] = 100
    data["SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV"] = 100
    data["SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] *= 0.0
    data["SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"] *= 0.0
    for step in range(obs_steps):
        data.loc[dates[step], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] += 0.2 * step
        data.loc[dates[step], "SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"] += 0.2 * step

    data["SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"] = 0
    data["SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"] = 0

    return data

def test_P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_early_in_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.zeros((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["1/22/2025 10:00"]))
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 22
    assert pf.data["day_of_week"].iloc[0] == 2
    assert pf.data["time_of_day"].iloc[0] == 36000

    out = virtual_tag_stage(pf)

    # '(100 - {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV} = 0
    # {day_of_year} = 22
    # (1 / (1 + 2.71828**(-1*((22 - 250) / 20)))) = 0.000011
    # (100 - 0) * 0.000011 = 0.0011
    assert np.isclose(out.data["P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT"], 0.0011, atol=1e-4)

def test_P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_mid_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.ones((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["7/13/2025 16:00"]))
    df["SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV"] = 30.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 194
    assert pf.data["day_of_week"].iloc[0] == 6
    assert pf.data["time_of_day"].iloc[0] == 57600

    out = virtual_tag_stage(pf)

    # '(100 - {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV} = 30
    # {day_of_year} = 194
    # (1 / (1 + 2.71828**(-1*((194 - 250) / 20)))) = 0.057324
    # (100 - 30) * 0.057324 = 4.01268
    assert np.isclose(out.data["P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT"], 4.01268, atol=1e-4)

def test_P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_end_of_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.ones((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["11/29/2025 12:00"]))
    df["SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV"] = 45.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 333
    assert pf.data["day_of_week"].iloc[0] == 5
    assert pf.data["time_of_day"].iloc[0] == 43200

    out = virtual_tag_stage(pf)

    # '(100 - {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV} = 45
    # {day_of_year} = 333
    # (1 / (1 + 2.71828**(-1*((333 - 250) / 20)))) = 0.98448
    # (100 - 45) * 0.98448 = 54.1464
    assert np.isclose(out.data["P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT"], 54.1464, atol=1e-4)

def test_P72_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT_early_in_year(
    cfg: MainConfig,
    app_state: AppState,
    tag_names: list[str],
):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    df = pd.DataFrame(data=np.ones((1,len(tag_names))), columns=tag_names, index=pd.DatetimeIndex(["2/14/2025 11:00"]))
    df["SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV"] = 85.7
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 45
    assert pf.data["day_of_week"].iloc[0] == 4
    assert pf.data["time_of_day"].iloc[0] == 39600

    out = virtual_tag_stage(pf)

    # '(100 - {SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV} = 85.7
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
    df["SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV"] = 20.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 175
    assert pf.data["day_of_week"].iloc[0] == 1
    assert pf.data["time_of_day"].iloc[0] == 70020

    out = virtual_tag_stage(pf)

    # '(100 - {SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV} = 20
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
    df["SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV"] = 50.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    assert "day_of_year" in pf.data.columns
    assert "day_of_week" in pf.data.columns
    assert "time_of_day" in pf.data.columns
    assert pf.data["day_of_year"].iloc[0] == 345
    assert pf.data["day_of_week"].iloc[0] == 3
    assert pf.data["time_of_day"].iloc[0] == 21600

    out = virtual_tag_stage(pf)

    # '(100 - {SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # {SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV} = 50
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
    df["SCADA1.ELS_SLR_ENG_BATT_COST_SP.F_CV"] = 5.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV"] = 105.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"] = 100.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV"] = 111.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV"] = 111.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"] = 111.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"] = 117.0
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV"].iloc[1], 5.0)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"].iloc[1], 0.0)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV"].iloc[1], 0.0)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"].iloc[1], 6.0)

    out = virtual_tag_stage(pf)

    # '{SCADA1.ELS_SLR_ENG_BATT_COST_SP.F_CV} * ({SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV} + {SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV} + {SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV} + {SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV})'
    # 5 * (5 + 0 + 0 + 6) = 55
    assert np.isnan(out.data["BESS_DEGRADATION_COST"].iloc[0])
    assert np.isclose(out.data["BESS_DEGRADATION_COST"].iloc[1], 55.0)

def test_ENERGY_COST(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05"])
    df = pd.DataFrame(data=np.ones((2, len(tag_names))), columns=tag_names, index=dates)
    df["SCADA1.ELS_SLR_ENG_IMPORT_COST_SP.F_CV"] = 47.42
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] = 105.0
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"] = 288.8
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"] = 294.7
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"].iloc[1], 5.0)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"].iloc[1], 5.9)

    out = virtual_tag_stage(pf)

    # '{SCADA1.ELS_SLR_ENG_IMPORT_COST_SP.F_CV} * ({SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV} + {SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV})'
    # 47.42 * (5 + 5.9)
    assert np.isnan(out.data["ENERGY_COST"].iloc[0])
    assert np.isclose(out.data["ENERGY_COST"].iloc[1], 516.878)

def test_ENERGY_SOLD(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05"])
    df = pd.DataFrame(data=np.ones((2, len(tag_names))), columns=tag_names, index=dates)
    df["Actual_Posted_Pool_Price"] = 111.11
    df["SCADA1.ELS_SLR_ENG_EXPORT_FEE_SP.F_CV"] = 0.375
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV"] = 105.0
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV"] = 288.8
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV"] = 294.7
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV"].iloc[1], 5.0)
    assert np.isnan(pf.data["SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV"].iloc[0])
    assert np.isclose(pf.data["SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV"].iloc[1], 5.9)

    out = virtual_tag_stage(pf)

    # '({Actual_Posted_Pool_Price} - {SCADA1.ELS_SLR_ENG_EXPORT_FEE_SP.F_CV}) * ({SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV} + {SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV})'
    # (111.11 - 0.375) * (5 + 5.9) = 1207.0115
    assert np.isnan(out.data["ENERGY_SOLD"].iloc[0])
    assert np.isclose(out.data["ENERGY_SOLD"].iloc[1], 1207.0115)

def test_PROFIT(cfg: MainConfig, app_state: AppState, tag_names: list[str]):
    seasonal_tag_stage = SeasonalTagIncluder(cfg.pipeline.tags)
    delta_tag_stage = DeltaizeTags(cfg.pipeline.tags, cfg.pipeline.delta)
    virtual_tag_stage = VirtualTagComputer(cfg.pipeline.tags, app_state)

    dates = pd.DatetimeIndex(["12/11/2025 6:00", "12/11/2025 6:05"])
    df = pd.DataFrame(data=np.ones((2, len(tag_names))), columns=tag_names, index=dates)
    df["SCADA1.ELS_SLR_ENG_BATT_COST_SP.F_CV"] = 5.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS1_ENG_IN_TOT.F_CV"] = 105.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS1_ENG_OUT_TOT.F_CV"] = 100.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV"] = 111.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS2_ENG_IN_TOT.F_CV"] = 111.0
    df.loc[dates[0], "SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"] = 111.0
    df.loc[dates[1], "SCADA1.ELS_SLR_BESS2_ENG_OUT_TOT.F_CV"] = 117.0
    df["SCADA1.ELS_SLR_ENG_IMPORT_COST_SP.F_CV"] = 47.42
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] = 105.0
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"] = 288.8
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP72_ENG_IN_TOT.F_CV"] = 294.7
    df["Actual_Posted_Pool_Price"] = 111.11
    df["SCADA1.ELS_SLR_ENG_EXPORT_FEE_SP.F_CV"] = 0.375
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV"] = 100.0
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP12_ENG_OUT_TOT.F_CV"] = 105.0
    df.loc[dates[0], "SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV"] = 288.8
    df.loc[dates[1], "SCADA1.ELS_SLR_ELSP72_ENG_OUT_TOT.F_CV"] = 294.7
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = seasonal_tag_stage(pf)
    pf = delta_tag_stage(pf)

    out = virtual_tag_stage(pf)

    # '{ENERGY_SOLD} - {ENERGY_COST} - {BESS_DEGRADATION_COST}'
    # 1207.0115 - 516.878 - 55.0 = 635.1335
    assert np.isnan(out.data["PROFIT"].iloc[0])
    assert np.isclose(out.data["PROFIT"].iloc[1], 635.1335)

def test_no_violations(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    obs_steps = int(cfg.interaction.action_period / cfg.interaction.obs_period) + 1

    pipeline = Pipeline(app_state, cfg.pipeline)

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * (0.1667 + 0.1667) = 1.667
    # ENERGY_COST = 47.42 * (0.2 + 0.2) = 18.968
    # ENERGY_SOLD = (111.11 - 0.375) * (0.5 + 0.5) = 110.735
    # PROFIT = 110.735 - 18.968 - 1.667 = 90.1
    # Normalized PROFIT = 0.237586
    # Scaled to [-1, 0]: -0.762414
    # With Return Normalization: -0.00762414
    # n-step reward: -0.00762414
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == obs_steps
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.00762414, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.00762414, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.762414, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.762414, atol=1e-4)

def test_curtailment_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    obs_steps = int(cfg.interaction.action_period / cfg.interaction.obs_period) + 1

    pipeline = Pipeline(app_state, cfg.pipeline)

    # Red/Yellow Zone Tags
    df["SCADA1.ELS_SLR_ELSP12_KW.F_CV"] = 4750 # Curtailment yellow zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * (0.1667 + 0.1667) = 1.667
    # ENERGY_COST = 47.42 * (0.2 + 0.2) = 18.968
    # ENERGY_SOLD = (111.11 - 0.375) * (0.5 + 0.5) = 110.735
    # PROFIT = 110.735 - 18.968 - 1.667 = 90.1
    # Normalized PROFIT = 0.237586
    # Scaled to [-1, 0]: -0.762414
    # SCADA1.ELS_SLR_ELSP12_KW.F_CV = 4750 represents 16.66% yellow zone violation
    # Penalty = -2 * (0.1666 ** 2) = -0.05551112
    # Unnormalized Reward: -0.762414 + -0.05551112 = -0.81792512
    # With Return Normalization: -0.0081792512
    # n-step reward: -0.0081792512
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == obs_steps
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.0081792512, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.0081792512, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.81792512, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.81792512, atol=1e-4)

def test_power_peak_yellow_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    obs_steps = int(cfg.interaction.action_period / cfg.interaction.obs_period) + 1
    dates = df.index.tolist()

    pipeline = Pipeline(app_state, cfg.pipeline)

    # Red/Yellow Zone Tags
    # Red Bound: '({SCADA1.ELS_SLR_ELSP72_POWER_PEAK_SP_LGC.F_CV} / 1000) * 5 / 60' = (4400 / 1000) * 5 / 60 = 0.36667
    # Yellow Bound: 0.95 * 0.36667 = 0.3483
    df["SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] = 0.0
    for step in range(obs_steps):
        df.loc[dates[step], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] += 0.35 * step # yellow zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * (0.1667 + 0.1667) = 1.667
    # ENERGY_COST = 47.42 * (0.35 + 0.2) = 26.081
    # ENERGY_SOLD = (111.11 - 0.375) * (0.5 + 0.5) = 110.735
    # PROFIT = 110.735 - 26.081 - 1.667 = 82.987
    # Normalized PROFIT = 0.2352636
    # Scaled to [-1, 0]: -0.7647364
    # SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV = 0.35 represents 9.0909% yellow zone violation
    # Penalty = -2 * (0.090909 ** 2) = -0.01652889
    # Unnormalized Reward: -0.7647364 + -0.01652889 = -0.78126529
    # With Return Normalization: -0.0078126529
    # n-step reward: -0.0078126529
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == obs_steps
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.0078126529, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.0078126529, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.78126529, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.78126529, atol=1e-4)

def test_power_peak_red_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    obs_steps = int(cfg.interaction.action_period / cfg.interaction.obs_period) + 1
    dates = df.index.tolist()

    pipeline = Pipeline(app_state, cfg.pipeline)

    # Red/Yellow Zone Tags
    # Red Bound: '({SCADA1.ELS_SLR_ELSP72_POWER_PEAK_SP_LGC.F_CV} / 1000) * 5 / 60' = (4400 / 1000) * 5 / 60 = 0.36667
    # Yellow Bound: 0.95 * 0.36667 = 0.3483
    df["SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] = 0.0
    for step in range(obs_steps):
        df.loc[dates[step], "SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV"] += 0.4 * step  # red zone violation

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # SCADA1.ELS_SLR_ELSP12_ENG_IN_TOT.F_CV = 0.4 represents 5.2631% red zone violation
    # Unnormalized Reward = -4 - (4 * 0.052626) = -4.210504
    # With Return Normalization: -0.04210504
    # n-step reward: -0.04210504
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == obs_steps
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
    action_period = cfg.interaction.action_period
    obs_steps = int(action_period / obs_period) + 1
    start = datetime.datetime(2025, 12, 22, 10)
    dates = [start + i * obs_period for i in range(obs_steps)]
    df.index = dates

    # Red/Yellow Zone Tags
    df["SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV"] = 50 # On-site consumption yellow zone violation
    df["SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV"] = 52

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # BESS_DEGRADATION_COST = 5 * (0.1667 + 0.1667) = 1.667
    # ENERGY_COST = 47.42 * (0.2 + 0.2) = 18.968
    # ENERGY_SOLD = (111.11 - 0.375) * (0.5 + 0.5) = 110.735
    # PROFIT = 110.735 - 18.968 - 1.667 = 90.1
    # Normalized PROFIT = 0.237586
    # Scaled to [-1, 0]: -0.762414
    # P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT:
    # '(100 - {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # (100 - 50) * (1 / (1 + 2.71828**(-1*((356 - 250) / 20)))) = 49.751659
    # Represents 97.31438% yellow zone violation
    # Penalty = -2 * (0.97314 ** 2) = -1.894
    # Unnormalized Reward: -0.762414 + -1.894 = -2.656414
    # With Return Normalization: -0.02656414
    # n-step reward: -0.02656414
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == obs_steps
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.02656414, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.02656414, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -2.656414, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -2.656414, atol=1e-4)

def test_on_site_consumption_red_zone_violation(cfg: MainConfig, app_state: AppState, df: pd.DataFrame):
    pipeline = Pipeline(app_state, cfg.pipeline)

    obs_period = cfg.interaction.obs_period
    action_period = cfg.interaction.action_period
    obs_steps = int(action_period / obs_period) + 1
    start = datetime.datetime(2025, 12, 22, 10)
    dates = [start + i * obs_period for i in range(obs_steps)]
    df.index = dates

    # Red/Yellow Zone Tags
    df["SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV"] = 48 # On-site consumption red zone violation
    df["SCADA1.ELS_SLR_ELSP72_MAJORITY_LGC.F_CV"] = 52

    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # Expected
    # P12_MAJORITY_ON_SITE_CONSUMPTION_CONSTRAINT:
    # '(100 - {SCADA1.ELS_SLR_ELSP12_MAJORITY_LGC.F_CV}) * (1 / (1 + 2.71828**(-1*(({day_of_year} - 250) / 20))))'
    # (100 - 48) * (1 / (1 + 2.71828**(-1*((356 - 250) / 20)))) = 51.7417
    # Represents 3.867978% red zone violation
    # Unnormalized Reward: -4 - (4 * 0.0386797) = -4.1547188
    # With Return Normalization: -0.041547188
    # n-step reward: -0.041547188
    # n-step gamma: 0.99

    assert len(got.transitions) == 1
    assert len(got.transitions[0].steps) == obs_steps
    assert np.isclose(got.transitions[0].n_step_gamma, 0.99, atol=1e-4)
    if cfg.feature_flags.normalize_return:
        assert np.isclose(got.transitions[0].steps[1].reward, -0.041547188, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -0.041547188, atol=1e-4)
    else:
        assert np.isclose(got.transitions[0].steps[1].reward, -4.1547188, atol=1e-4)
        assert np.isclose(got.transitions[0].n_step_reward, -4.1547188, atol=1e-4)
