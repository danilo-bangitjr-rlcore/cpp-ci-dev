import gymnasium as gym
import gym_electric_motor as gem
from gymnasium.wrappers import FlattenObservation
from root.environment.three_tanks import ThreeTankEnv
from root.environment.three_tanks import TTChangeAction, TTChangeActionDiscrete
from root.environment.smpl.envs.atropineenv import AtropineEnvGym
from root.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from root.environment.smpl.envs.reactorenv import ReactorEnvGym
from root.environment.gym_wrapper import DiscreteControlWrapper, D4RLWrapper
from root.environment.pendulum_env import PendulumEnv
from root.environment.reseau_env import ReseauEnv
from root.environment.influx_opc_env import DBClientWrapperBase
from root.environment.opc_connection import OpcConnection
from root.environment.wrapper.one_hot_wrapper import OneHotWrapper

import json


def init_environment(cfg):
    seed = cfg.seed
    name = cfg.name
    if name == "ThreeTank":
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

    elif name == "Reseau_online":
        db_settings = json.load(open(cfg.db_settings_pth, "r"))
        opc_settings = json.load(open(cfg.opc_settings_pth, "r"))

        db_client = DBClientWrapperBase(db_settings["bucket"], db_settings["org"],
                                        db_settings["token"], db_settings["url"])

        opc_connection = OpcConnection(opc_settings["IP"], opc_settings["port"])

        control_tag_default = [cfg.reset_fpm]
        runtime = None
        env = ReseauEnv(db_client, opc_connection, cfg.control_tags, control_tag_default, cfg.col_names, runtime,
                        obs_freq=cfg.obs_freq, obs_window=cfg.obs_window, last_n_obs=cfg.last_n_obs)

    else:
        raise NotImplementedError

    if cfg.discrete_control:
        env = OneHotWrapper(env)

    return env
