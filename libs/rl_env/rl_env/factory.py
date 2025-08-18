import logging
from typing import Annotated

from pydantic import Field

from rl_env.bsm1.bsm1 import BSM1Config
from rl_env.calibration import CalibrationConfig
from rl_env.distraction_world import DistractionWorldConfig
from rl_env.four_rooms import FourRoomsConfig
from rl_env.group_util import env_group
from rl_env.mountain_car import MountainCarContinuousConfig
from rl_env.multi_action_saturation import MultiActionSaturationConfig
from rl_env.pertube_env import ObservationPerturbationWrapper, PerturbationConfig
from rl_env.pvs import PVSConfig
from rl_env.saturation import SaturationConfig
from rl_env.saturation_goals import SaturationGoalsConfig
from rl_env.stand_still_mc import StandStillMCConfig
from rl_env.streamline import PipelineEnvConfig
from rl_env.t_maze import TMazeConfig
from rl_env.three_tanks import ThreeTanksConfig
from rl_env.windy_room import WindyRoomConfig

EnvConfig = Annotated[
    BSM1Config |
    CalibrationConfig |
    DistractionWorldConfig |
    FourRoomsConfig |
    MultiActionSaturationConfig |
    PVSConfig |
    SaturationConfig |
    SaturationGoalsConfig |
    StandStillMCConfig |
    PipelineEnvConfig |
    TMazeConfig |
    ThreeTanksConfig |
    WindyRoomConfig |
    MountainCarContinuousConfig,
    Field(discriminator="name"),
]

logger = logging.getLogger(__name__)

def init_env(cfg: EnvConfig, perturbation_config: dict | None = None):
    logger.info(f"instantiating {cfg.name} with config {cfg}")

    env = env_group.dispatch(cfg.name, cfg_obj=cfg)
    if perturbation_config:
        perturb_cfg = PerturbationConfig(**perturbation_config)
        env = ObservationPerturbationWrapper(env, perturb_cfg)
    return env
