import gymnasium as gym
import gym_electric_motor as gem
from gymnasium.wrappers import FlattenObservation
from src.environment.three_tanks import ThreeTankEnv, NonContexTT
from src.environment.three_tanks import TTChangeAction, TTAction, TTChangeActionDiscrete
from src.environment.three_tanks import TTChangeActionClip, TTChangeActionDiscreteClip
from src.environment.smpl.envs.atropineenv import AtropineEnvGym
from src.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from src.environment.smpl.envs.reactorenv import ReactorEnvGym
from src.environment.gym_wrapper import DiscreteControlWrapper
from src.environment.PendulumEnv import PendulumEnv


def init_environment(name, cfg):
    if name == "ThreeTank":
        return ThreeTankEnv(cfg.seed, cfg.lr_constrain, env_action_scaler=cfg.env_action_scaler, random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/ConstPID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler,
                              agent_action_min=0*cfg.action_scale+cfg.action_bias,
                              agent_action_max=1*cfg.action_scale+cfg.action_bias,
                              random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/ChangePID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=False, env_action_scaler=cfg.env_action_scaler,
                              agent_action_min=0*cfg.action_scale+cfg.action_bias,
                              agent_action_max=1*cfg.action_scale+cfg.action_bias,
                              random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/DiscreteConstPID":
        return TTChangeActionDiscrete(cfg.env_info[0], cfg.seed, cfg.lr_constrain, constant_pid=True,
                                      env_action_scaler=cfg.env_action_scaler, random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/DiscreteRwdStay":
        return TTChangeActionDiscrete(cfg.env_info[0], cfg.seed, cfg.lr_constrain, constant_pid=True,
                                      env_action_scaler=cfg.env_action_scaler, random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/ClipConstPID":
        return TTChangeActionClip(cfg.seed, cfg.lr_constrain, constant_pid=True,
                                  env_action_scaler=cfg.env_action_scaler,
                                  agent_action_min=0*cfg.action_scale+cfg.action_bias,
                                  agent_action_max=1*cfg.action_scale+cfg.action_bias,
                                  random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/ClipDiscreteConstPID":
        return TTChangeActionDiscreteClip(cfg.env_info[0], cfg.seed, cfg.lr_constrain, constant_pid=True,
                                          env_action_scaler=cfg.env_action_scaler, random_sp=cfg.env_info[1:])
    elif name == "TTAction/ConstPID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler, random_sp=cfg.env_info[1:])
    elif name == "TTAction/ChangePID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=False, env_action_scaler=cfg.env_action_scaler, random_sp=cfg.env_info[1:])
    elif name == "NonContexTT":
        return NonContexTT(cfg.seed, cfg.lr_constrain, env_action_scaler=cfg.env_action_scaler, obs=cfg.env_info[0])
    elif name == "AtropineEnv":
        return AtropineEnvGym()
    elif name == "BeerEnv":
        return BeerFMTEnvGym()
    elif name == "ReactorEnv":
        return ReactorEnvGym()
    elif name == "Cont-CC-PermExDc-v0":
        return FlattenObservation(gem.make("Cont-CC-PermExDc-v0"))
    elif name == "Cont-CC-PMSM-v0":
        return FlattenObservation(gem.make("Cont-CC-PMSM-v0"))
    elif name == "Cont-CC-DFIM-v0":
        return FlattenObservation(gem.make("Cont-CC-DFIM-v0"))
    elif name == "Cont-CC-SCIM-v0":
        return FlattenObservation(gem.make("Cont-CC-SCIM-v0"))
    elif name == "Cont-CC-EESM-v0":
        return FlattenObservation(gem.make("Cont-CC-EESM-v0"))
    elif name == "Acrobot-v1":
        return DiscreteControlWrapper("Acrobot-v1")
    elif name == "MountainCarContinuous-v0":
        return  gym.make("MountainCarContinuous-v0")
    elif name == "Pendulum-v1":
        return PendulumEnv()
    elif name == "HalfCheetah-v4":
        return gym.make("HalfCheetah-v4")
    else:
        raise NotImplementedError

def configure_action_scaler_and_bias(cfg):
    if cfg.auto_calibrate_beta_support:
        # auto scales based on gym env.action_space attributes
        if cfg.actor == 'Beta':
            action_low = cfg.train_env.action_space.low
            action_high = cfg.train_env.action_space.high
            action_range = action_high - action_low
            cfg.action_scale = action_range
            cfg.action_bias = action_low 
        elif cfg.actor == 'SGaussian':
            action_low = cfg.train_env.action_space.low
            action_high = cfg.train_env.action_space.high
            action_range = action_high - action_low
            cfg.action_scale = action_range / 2 # since SGaussian defined on [-1, 1]
            cfg.action_bias = action_low + cfg.action_scale 
    else: 
        # if we are not automatically calibrating the scale and bias based on the environment. 
        # We can set values here based on domain knowlegde
        name = cfg.env_name
        if name == "ThreeTank":
            if cfg.actor == 'Beta':
                cfg.action_scale = 10
                cfg.action_bias = 0
            elif cfg.actor == 'SGaussian':
                cfg.action_scale = 5
                cfg.action_bias = 5
            else:
                raise NotImplementedError
        elif name == "TTChangeAction/ConstPID":
            if cfg.actor == 'Beta':
                cfg.action_scale = 0.2
                cfg.action_bias = -0.1
            else:
                raise NotImplementedError
        elif name == "TTChangeAction/ChangePID":
            raise NotImplementedError
        elif name == "TTChangeAction/DiscreteConstPID":
            raise NotImplementedError
        elif name == "TTChangeAction/DiscreteRwdStay":
           raise NotImplementedError
        elif name == "TTChangeAction/ClipConstPID":
           raise NotImplementedError
        elif name == "TTChangeAction/ClipDiscreteConstPID":
            raise NotImplementedError
        elif name == "TTAction/ConstPID":
           raise NotImplementedError
        elif name == "TTAction/ChangePID":
           raise NotImplementedError
        elif name == "NonContexTT":
            if cfg.actor == 'Beta':
                cfg.action_scale = 10
                cfg.action_bias = 0
            elif cfg.actor == 'SGaussian':
                cfg.action_scale = 5
                cfg.action_bias = 5
        elif name == "AtropineEnv":
            raise NotImplementedError
        elif name == "BeerEnv":
            raise NotImplementedError
        elif name == "ReactorEnv":
            raise NotImplementedError
        elif name == "Cont-CC-PermExDc-v0":
            raise NotImplementedError
        elif name == "Cont-CC-PMSM-v0":
            raise NotImplementedError
        elif name == "Cont-CC-DFIM-v0":
            raise NotImplementedError
        elif name == "Cont-CC-SCIM-v0":
            raise NotImplementedError
        elif name == "Acrobot-v1":
            raise NotImplementedError
        elif name == "MountainCar-v0":
            raise NotImplementedError
        elif name == "Pendulum-v1":
            raise NotImplementedError
        else:
            raise NotImplementedError
    