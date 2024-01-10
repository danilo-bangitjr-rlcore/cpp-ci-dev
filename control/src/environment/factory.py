import gymnasium as gym
import gym_electric_motor as gem
from gymnasium.wrappers import FlattenObservation
from src.environment.three_tanks import ThreeTankEnv, NonContexTT
from src.environment.three_tanks import TTChangeAction, TTAction, TTChangeActionDiscrete
from src.environment.three_tanks import TTChangeActionClip, TTChangeActionDiscreteClip
from src.environment.smpl.envs.atropineenv import AtropineEnvGym
from src.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from src.environment.smpl.envs.reactorenv import ReactorEnvGym


def init_environment(name, cfg):
    if name == "ThreeTank":
        return ThreeTankEnv(cfg.seed, cfg.lr_constrain, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTChangeAction/ConstPID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler,
                              agent_action_min=0*cfg.action_scale+cfg.action_bias,
                              agent_action_max=1*cfg.action_scale+cfg.action_bias)
    elif name == "TTChangeAction/ChangePID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=False, env_action_scaler=cfg.env_action_scaler,
                              agent_action_min=0*cfg.action_scale+cfg.action_bias,
                              agent_action_max=1*cfg.action_scale+cfg.action_bias)
    elif name == "TTChangeAction/DiscreteConstPID":
        return TTChangeActionDiscrete(cfg.env_info, cfg.seed, cfg.lr_constrain, constant_pid=True,
                                      env_action_scaler=cfg.env_action_scaler)
    elif name == "TTChangeAction/DiscreteRwdStay":
        return TTChangeActionDiscrete(cfg.env_info, cfg.seed, cfg.lr_constrain, constant_pid=True,
                                      env_action_scaler=cfg.env_action_scaler, )
    elif name == "TTChangeAction/ClipConstPID":
        return TTChangeActionClip(cfg.seed, cfg.lr_constrain, constant_pid=True,
                                  env_action_scaler=cfg.env_action_scaler,
                                  agent_action_min=0*cfg.action_scale+cfg.action_bias,
                                  agent_action_max=1*cfg.action_scale+cfg.action_bias)
    elif name == "TTChangeAction/ClipDiscreteConstPID":
        return TTChangeActionDiscreteClip(cfg.env_info, cfg.seed, cfg.lr_constrain, constant_pid=True,
                                          env_action_scaler=cfg.env_action_scaler)
    elif name == "TTAction/ConstPID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTAction/ChangePID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=False, env_action_scaler=cfg.env_action_scaler)
    elif name == "NonContexTT":
        return NonContexTT(cfg.seed, cfg.lr_constrain, env_action_scaler=cfg.env_action_scaler)
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
    elif name == "Acrobot-v0":
        return gym.make("Acrobot-v0")
    elif name == "MountainCar-v0":
        gym.make("MountainCar-v0")
    elif name == "Pendulum-v1":
        gym.make("Pendulum-v1")
    else:
        raise NotImplementedError

