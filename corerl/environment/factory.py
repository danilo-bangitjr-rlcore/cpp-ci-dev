import logging
from typing import Any

import gymnasium as gym
import numpy as np

import corerl.environment.three_tanks as TTv1
import corerl.environment.three_tanks_v2 as TTv2
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.environment.bimodal import Bimodal
from corerl.environment.delayed_saturation import DelayedSaturation
from corerl.environment.four_rooms import FourRoomsEnv
from corerl.environment.reseau_env import ReseauEnv
from corerl.environment.saturation import Saturation
from corerl.environment.smpl.envs.atropineenv import AtropineEnvGym
from corerl.environment.smpl.envs.beerfmtenv import BeerFMTEnvGym
from corerl.environment.smpl.envs.reactorenv import ReactorEnvGym
from corerl.environment.wrapper.discrete_control_wrapper import DiscreteControlWrapper, SparseDiscreteControlWrapper
from corerl.environment.wrapper.one_hot_wrapper import OneHotWrapper

log = logging.getLogger(__name__)


def init_environment(cfg: AsyncEnvConfig) -> gym.Env:
    seed = cfg.seed
    name = cfg.name

    # NOTE: not all of these wrappers and envs actually inherit `gym.Env`
    # which means the type of env cannot be `gym.Env`. We could use a
    # Prototype and spell out the desired contract, or we could fix
    # downstream implementations to inherit `gym.Env`.
    env: Any | None = None

    # proposed syntax, env defined directly in config
    args = cfg.args
    kwargs = cfg.kwargs
    if isinstance(cfg, DepAsyncEnvConfig) and (bool(args) or bool(kwargs)):
        log.warning("Environment args or kwargs set in confinguration for DeploymentAsyncEnv. This will be ignored!")
        args = []
        kwargs = {}
    return gym.make(cfg.gym_name, *args, **kwargs)

    # prior syntax, lookup based on config.name
    if name == "three_tanks_v2":
        if cfg.discrete_control:
            raise NotImplementedError

        if cfg.change_action:
            temp = cfg.reset_temperature
            if temp in ("np.inf", "∞", "inf"):
                temp = np.inf

            env = TTv2.TTChangeAction(
                seed,
                mse_penalty_reward=cfg.mse_penalty_reward,
                reset_to_high_reward=cfg.reset_to_high_reward,
                reset_buffer_size=cfg.reset_buffer_size,
                reset_buffer_add_delay=cfg.reset_buffer_add_delay,
                reset_buffer_always_include_start=cfg.reset_buffer_always_include_start,
                reset_temperature=temp,
                n_internal_iter=cfg.n_internal_iter,
            )
        else:
            env = TTv2.ThreeTankEnv(seed, random_sp=cfg.random_sp)

    elif name == "four_rooms":
        env = FourRoomsEnv(
            seed,
            action_scale=cfg.action_scale,
            noise_scale=cfg.noise_scale,
            decay_scale=cfg.decay_scale,
            decay_probability=cfg.decay_probability,
            continuous_action=not cfg.discrete_control,
        )

    elif name == "three_tanks":
        if cfg.change_action:
            if cfg.discrete_action:
                TTv1.TTChangeActionDiscrete(
                    cfg.delta_step, seed, cfg.lr_constrain,
                    constant_pid=cfg.constant_pid, random_sp=cfg.random_sp,
                )
            else:
                env = TTv1.TTChangeAction(
                    seed, cfg.lr_constrain, constant_pid=cfg.constant_pid,
                    agent_action_min=-1, agent_action_max=1,
                    random_sp=cfg.random_sp,
                )
        else:
            if cfg.discrete_control:
                raise NotImplementedError
            else:
                env = TTv1.ThreeTankEnv(
                    seed, cfg.lr_constrain, random_sp=cfg.random_sp,
                )
    elif name == "AtropineEnv":
        # remove timeout in environment, set timeout in the interaction layer
        env = AtropineEnvGym(max_steps=-1)
    elif name == "BeerEnv":
        env = BeerFMTEnvGym(max_steps=-1)
    elif name == "ReactorEnv":
        env = ReactorEnvGym(max_steps=-1)
    # unsure about these gem things
    # Didn't find interface for timeout & seed setting for gem environments.
    elif name == "Acrobot-v1":
        env = DiscreteControlWrapper("Acrobot-v1", seed)
    elif name == "Acrobot-v1-sparse":
        env = SparseDiscreteControlWrapper("Acrobot-v1", seed)
    elif name == "MountainCarContinuous-v0":
        env = gym.make("MountainCarContinuous-v0")
        env._max_episode_steps = np.inf # type: ignore
        env.reset(seed=seed)
    elif name == "MountainCar-v0":
        env = DiscreteControlWrapper("MountainCar-v0", seed)
    elif name == "MountainCar-v0-sparse":
        env = SparseDiscreteControlWrapper("MountainCar-v0", seed)
    elif name == "Bimodal":
        reward_v = False
        if "reward_variance" in cfg.keys():
            reward_v = cfg.reward_variance
        env = Bimodal(seed, reward_v)
    elif name == "HalfCheetah-v4":
        env = gym.make("HalfCheetah-v4")
        env._max_episode_steps = np.inf # type: ignore
        env.reset(seed=seed)
    elif name == 'reseau':
        env = ReseauEnv(cfg)
    elif name == 'saturation':
        env = Saturation()
    elif name == 'delayed_saturation':
        env = DelayedSaturation(seed, cfg)
    else:
        raise NotImplementedError

    assert env is not None
    if cfg.discrete_control:
        env = OneHotWrapper(env)

    return env
