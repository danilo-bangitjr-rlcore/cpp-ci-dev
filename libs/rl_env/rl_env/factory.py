import logging
import warnings

from rl_env.group_util import env_group
from rl_env.pertube_env import ObservationPerturbationWrapper, PerturbationConfig

logger = logging.getLogger(__name__)

def init_env(name: str, overrides: dict | None = None, perturbation_config: dict | None = None):
    # register environments
    import rl_env.bsm1
    import rl_env.calibration
    import rl_env.distraction_world
    import rl_env.four_rooms
    import rl_env.multi_action_saturation
    import rl_env.pvs
    import rl_env.saturation
    import rl_env.saturation_goals
    import rl_env.stand_still_mc
    import rl_env.t_maze
    import rl_env.three_tanks
    import rl_env.windy_room  # noqa: F401

    logger.info(f"instantiating {name} with overrides {overrides}")

    if name == 'DelayedSaturation-v0':
        warnings.warn(
            "Delayed-Saturation-v0 is deprecated. Use Saturation-v0 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = 'Saturation-v0'
        overrides = {
            'decay': 0.75,
            'effect': None,
            'effect_period': 500,
            'trace_val': 0.9,
        }

    env = env_group.dispatch(name, overrides)
    if perturbation_config:
        perturb_cfg = PerturbationConfig(**perturbation_config)
        env = ObservationPerturbationWrapper(env, perturb_cfg)
    return env
