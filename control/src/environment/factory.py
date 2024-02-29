import gymnasium as gym
import gym_electric_motor as gem
from gymnasium.wrappers import FlattenObservation
from src.environment.three_tanks import ThreeTankEnv, NonContexTT
from src.environment.three_tanks import TTChangeAction, TTAction, TTChangeActionDiscrete
from src.environment.smpl.envs.atropineenv import AtropineEnvGym
from src.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from src.environment.smpl.envs.reactorenv import ReactorEnvGym
from src.environment.gym_wrapper import DiscreteControlWrapper, D4RLWrapper
from src.environment.PendulumEnv import PendulumEnv
from src.environment.ReseauEnv import ReseauEnv
from src.environment.InfluxOPCEnv import DBClientWrapperBase
from src.environment.opc_connection import OpcConnection

import json


def init_environment(name, cfg):
    if name == "ThreeTank":
        return ThreeTankEnv(cfg.seed, cfg.lr_constrain, random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/ConstPID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=True,
                              agent_action_min=-1,#0*cfg.action_scale+cfg.action_bias,
                              agent_action_max=1,#1*cfg.action_scale+cfg.action_bias,
                              random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/ChangePID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=False,
                              agent_action_min=-1, #0*cfg.action_scale+cfg.action_bias,
                              agent_action_max=1, #1*cfg.action_scale+cfg.action_bias,
                              random_sp=cfg.env_info[1:])
    elif name == "TTChangeAction/DiscreteConstPID":
        return TTChangeActionDiscrete(cfg.env_info[0], cfg.seed, cfg.lr_constrain, constant_pid=True,
                                      random_sp=cfg.env_info[1:])
    elif name == "TTAction/ConstPID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=True, random_sp=cfg.env_info[1:])
    elif name == "TTAction/ChangePID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=False, random_sp=cfg.env_info[1:])
    elif name == "NonContexTT":
        return NonContexTT(cfg.seed, cfg.lr_constrain, obs=cfg.env_info[0])
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
        return DiscreteControlWrapper("Acrobot-v1", cfg.timeout)
    elif name == "MountainCarContinuous-v0":
        return  gym.make("MountainCarContinuous-v0")
    elif name == "Pendulum-v1":
        if cfg.discrete_control:
            return PendulumEnv(render_mode="human", continuous_action=False) 
        else:
            return PendulumEnv(render_mode="human")
    elif name == "HalfCheetah-v4":
        return gym.make("HalfCheetah-v4")
    elif name == "Walker2d-expert":
        return D4RLWrapper("walker2d-expert-v2", cfg.seed)
    elif name == "Walker2d-medium":
        return D4RLWrapper("walker2d-medium-v2", cfg.seed)

    elif name == "Reseau_online":
        db_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\db_settings_osoyoos.json"
        db_settings = json.load(open(db_settings_pth, "r"))
        
        opc_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\opc_settings_osoyoos.json"
        opc_settings = json.load(open(opc_settings_pth, "r"))
        
        db_client = DBClientWrapperBase(db_settings["bucket"], db_settings["org"], 
                            db_settings["token"], db_settings["url"])
        
        opc_connection = OpcConnection(opc_settings["IP"], opc_settings["port"])

        control_tags = ["osoyoos.plc.Process_DB.P250 Flow Pace Calc.Flow Pace Multiplier"]
        control_tag_default = [cfg.reset_fpm]
        runtime = None
        col_names = [
            "ait101_pv",
            "ait301_pv",
            "ait401_pv",
            "fit101_pv",
            "fit210_pv",
            "fit230_pv",
            "fit250_pv",
            "fit401_pv", 
            "p250_fp", 
            "pt100_pv",
            "pt101_pv", 
            "pt161_pv"
            ]
        
        return ReseauEnv(db_client, opc_connection, control_tags, control_tag_default, col_names, runtime,
                  obs_freq=cfg.obs_freq, obs_window=cfg.obs_window, last_n_obs=cfg.last_n_obs)
            
    else:
        raise NotImplementedError

def configure_action_scaler_and_bias(cfg):
    if cfg.auto_calibrate_beta_support:
        # auto scales based on gym env.action_space attributes
        if cfg.actor == 'Beta':
            action_low = cfg.train_env.action_space.low
            action_high = cfg.train_env.action_space.high
            action_range = action_high - action_low
            cfg.action_scale = float(action_range[0])
            cfg.action_bias = float(action_low[0]) 
        elif cfg.actor == 'SGaussian':
            action_low = cfg.train_env.action_space.low
            action_high = cfg.train_env.action_space.high
            action_range = action_high - action_low
            cfg.action_scale = float(action_range[0]) / 2 # since SGaussian defined on [-1, 1]
            cfg.action_bias = float(action_low[0]) + cfg.action_scale 
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
            cfg.state_normalizer = "TTChangeActionState"
            cfg.reward_normalizer = "ThreeTanksReward"
            if cfg.actor == 'Beta':
                cfg.action_scale = 10
                cfg.action_bias = -5
            else:
                raise NotImplementedError
        elif name == "TTChangeAction/ChangePID":
            raise NotImplementedError
        elif name == "TTChangeAction/DiscreteConstPID":
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
            cfg.state_normalizer = "Identity"
            cfg.reward_normalizer = "ThreeTanksReward"
            if cfg.actor == 'Beta':
                cfg.action_scale = 9.8
                cfg.action_bias = 0.2
            elif cfg.actor == 'SGaussian':
                cfg.action_scale = 4.9
                cfg.action_bias = 5.1
        elif name == "AtropineEnv":
            raise NotImplementedError
        elif name == "BeerEnv":
            raise NotImplementedError
        elif name == "ReactorEnv":
            if cfg.actor == 'Beta':
                cfg.action_scale = 2.0
                cfg.action_bias = -1.0
        elif name == "Cont-CC-PermExDc-v0":
            raise NotImplementedError
        elif name == "Cont-CC-PMSM-v0":
            raise NotImplementedError
        elif name == "Cont-CC-DFIM-v0":
            raise NotImplementedError
        elif name == "Cont-CC-SCIM-v0":
            raise NotImplementedError
        elif name == "Acrobot-v1":
            cfg.state_normalizer = "Identity"
            cfg.reward_normalizer = "Identity"
            cfg.action_normalizer = "OneHot"
            if cfg.actor == 'Softmax':
                cfg.action_scale = 3 # This is a bad naming. It should be the action dimension.
                cfg.action_bias = 0
            else:
                raise NotImplementedError
        elif name == "MountainCar-v0":
            raise NotImplementedError
        elif name == "Pendulum-v1":
            cfg.state_normalizer = "Identity"
            cfg.reward_normalizer = "Identity"
            cfg.action_normalizer = "OneHot"
            if cfg.actor == 'Softmax':
                cfg.action_scale = 3 # This is a bad naming. It should be the action dimension.
                cfg.action_bias = 0
            else:
                raise NotImplementedError
        elif name == "Reseau_online":
            if cfg.actor == 'Beta':
                cfg.action_scale = 100.0
                cfg.action_bias = 0.0
            else:
                raise NotImplementedError
        elif name in ["Walker2d-expert", "Walker2d-medium"]:
            if cfg.actor == 'Beta':
                cfg.action_scale = 100.0
                cfg.action_bias = 0.0
            else:
                raise NotImplementedError
        else:
            print(name, "configure not defined")
            raise NotImplementedError
    
    
