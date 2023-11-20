from src.environment.three_tanks import ThreeTankEnv, TTChangeAction, TTAction, TTChangeActionDiscrete
from src.environment.smpl.envs.atropineenv import AtropineEnvGym
from src.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from src.environment.smpl.envs.reactorenv import ReactorEnvGym


def init_environment(name, cfg):
    if name == "ThreeTank":
        return ThreeTankEnv(cfg.seed, cfg.lr_constrain)
    elif name == "TTChangeAction/ConstPID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=True)
    elif name == "TTChangeAction/ChangePID":
        return TTChangeAction(cfg.seed, cfg.lr_constrain, constant_pid=False)
    elif name == "TTChangeAction/DiscreteConstPID":
        return TTChangeActionDiscrete(cfg.env_info, cfg.seed, cfg.lr_constrain, constant_pid=True)
    elif name == "TTAction/ConstPID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=True)
    elif name == "TTAction/ChangePID":
        return TTAction(cfg.seed, cfg.lr_constrain, constant_pid=False)
    elif name == "AtropineEnv":
        return AtropineEnvGym()
    elif name == "BeerEnv":
        return BeerFMTEnvGym()
    elif name == "ReactorEnv":
        return ReactorEnvGym()
    else:
        raise NotImplementedError

