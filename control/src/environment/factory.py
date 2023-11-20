from src.environment.three_tanks import ThreeTankEnv, TTChangeAction, TTAction, TTChangeActionDiscrete
from src.environment.smpl.envs.atropineenv import AtropineEnvGym
from src.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from src.environment.smpl.envs.reactorenv import ReactorEnvGym


def init_environment(name, cfg):
    if name == "ThreeTank":
        return ThreeTankEnv(cfg.seed, cfg.lr_constrain, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTChangeAction/ConstPID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTChangeAction/ChangePID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=False, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTChangeAction/DiscreteConstPID":
        return TTChangeActionDiscrete(cfg.env_info, cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTAction/ConstPID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=True, env_action_scaler=cfg.env_action_scaler)
    elif name == "TTAction/ChangePID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=False, env_action_scaler=cfg.env_action_scaler)
    elif name == "AtropineEnv":
        return AtropineEnvGym()
    elif name == "BeerEnv":
        return BeerFMTEnvGym()
    elif name == "ReactorEnv":
        return ReactorEnvGym()
    else:
        raise NotImplementedError

