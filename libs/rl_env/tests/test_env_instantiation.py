import pytest

from rl_env.bsm1 import BSM1Config
from rl_env.calibration import CalibrationConfig
from rl_env.distraction_world import DistractionWorldConfig
from rl_env.factory import EnvConfig, init_env
from rl_env.four_rooms import FourRoomsConfig
from rl_env.multi_action_saturation import MultiActionSaturationConfig
from rl_env.pvs import PVSConfig
from rl_env.saturation import SaturationConfig
from rl_env.saturation_goals import SaturationGoalsConfig
from rl_env.stand_still_mc import StandStillMCConfig
from rl_env.t_maze import TMazeConfig
from rl_env.three_tanks import ThreeTanksConfig
from rl_env.windy_room import WindyRoomConfig

env_configs = [
    BSM1Config(),
    CalibrationConfig(),
    DistractionWorldConfig(),
    FourRoomsConfig(),
    MultiActionSaturationConfig(),
    PVSConfig(),
    SaturationConfig(),
    SaturationGoalsConfig(),
    StandStillMCConfig(),
    TMazeConfig(),
    ThreeTanksConfig(),
    WindyRoomConfig(),
]

@pytest.mark.parametrize('env_cfg', env_configs)
def test_env_instantiation(env_cfg: EnvConfig):
    env = init_env(env_cfg)
    assert env is not None

def test_saturation_env_with_overrides():
    cfg = SaturationConfig(
        decay=0.42,
        effect=0.99,
        effect_period=123,
        trace_val=0.77,
    )
    env = init_env(cfg)
    assert env is not None
