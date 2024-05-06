import gymnasium as gym
import gym_electric_motor as gem
from omegaconf import DictConfig
from gymnasium.wrappers import FlattenObservation

from corerl.environment.reseau_env import ReseauEnv
from corerl.environment.three_tanks import ThreeTankEnv
from corerl.environment.three_tanks import TTChangeAction, TTChangeActionDiscrete
from corerl.environment.smpl.envs.atropineenv import AtropineEnvGym
from corerl.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from corerl.environment.smpl.envs.reactorenv import ReactorEnvGym
from corerl.environment.wrapper.discrete_control_wrapper import DiscreteControlWrapper
from corerl.environment.wrapper.d4rl import D4RLWrapper
from corerl.environment.pendulum_env import PendulumEnv
from corerl.environment.wrapper.one_hot_wrapper import OneHotWrapper

def init_environment(cfg: DictConfig) -> gym.Env:
    seed = cfg.seed
    name = cfg.name
    if name == "three_tanks":
        if cfg.change_action:
            if cfg.discrete_action:
                TTChangeActionDiscrete(cfg.delta_step, seed, cfg.lr_constrain,
                                       constant_pid=cfg.constant_pid,
                                       random_sp=cfg.random_sp)
            else:
                env = TTChangeAction(seed, cfg.lr_constrain, constant_pid=cfg.constant_pid,
                                     agent_action_min=-1,
                                     agent_action_max=1,
                                     random_sp=cfg.random_sp)
        else:
            if cfg.discrete_control:
                raise NotImplementedError
            else:
                env = ThreeTankEnv(seed, cfg.lr_constrain, random_sp=cfg.random_sp)

    elif name == "AtropineEnv":
        env = AtropineEnvGym()
    elif name == "BeerEnv":
        env = BeerFMTEnvGym()
    elif name == "ReactorEnv":
        env = ReactorEnvGym()
    # unsure about these gem things
    elif name == "Cont-CC-PermExDc-v0":
        env = FlattenObservation(gem.make("Cont-CC-PermExDc-v0"))
    elif name == "Cont-CC-PMSM-v0":
        env = FlattenObservation(gem.make("Cont-CC-PMSM-v0"))
    elif name == "Cont-CC-DFIM-v0":
        env = FlattenObservation(gem.make("Cont-CC-DFIM-v0"))
    elif name == "Cont-CC-SCIM-v0":
        env = FlattenObservation(gem.make("Cont-CC-SCIM-v0"))
    elif name == "Cont-CC-EESM-v0":
        env = FlattenObservation(gem.make("Cont-CC-EESM-v0"))
    elif name == "Acrobot-v1":
        env = DiscreteControlWrapper("Acrobot-v1", cfg.timeout)
    elif name == "MountainCarContinuous-v0":
        env = gym.make("MountainCarContinuous-v0")
    elif name == "Pendulum-v1":
        env = PendulumEnv(render_mode="human", continuous_action=(not cfg.discrete_control))
    elif name == "HalfCheetah-v4":
        env = gym.make("HalfCheetah-v4")
    elif name == "Ant-expert":
        env = D4RLWrapper("ant-expert-v2", seed)
    elif name == "Walker2d-expert":
        env = D4RLWrapper("walker2d-expert-v2", seed)
    elif name == "Walker2d-medium":
        env = D4RLWrapper("walker2d-medium-v2", seed)
    elif name == 'reseau':
        env = ReseauEnv(cfg)
    else:
        raise NotImplementedError

    if cfg.discrete_control:
        env = OneHotWrapper(env)

    return env
