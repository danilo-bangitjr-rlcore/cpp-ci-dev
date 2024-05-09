import hydra
import numpy as np
import torch
import random

import hashlib
import copy

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces.utils import flatdim
from gymnasium import spaces
from pathlib import Path

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.eval.composite_eval import CompositeEval
from corerl.interaction.factory import init_interaction
from corerl.utils.device import init_device
from corerl.data_loaders.factory import init_data_loader
from corerl.environment.reward.factory import init_reward_function
from corerl.utils.plotting import make_plots

import corerl.utils.freezer as fr

"""
Revan: This is an example of how to run the code in the library.
I expect that each project may need something slightly different than what's
here.
"""


def prepare_save_dir(cfg):
    if cfg.experiment.param_from_hash:
        print("actor", cfg.agent.actor.actor_optimizer.lr)
        print("critic", cfg.agent.critic.critic_optimizer.lr)
        print("seed", cfg.experiment.seed)
        cfg_copy = copy.deepcopy(cfg)
        del cfg_copy.experiment.seed
        cfg_hash = hashlib.sha1(str(cfg_copy).encode("utf-8")).hexdigest()
        print("Using experiment param from hash:", cfg_hash)
        cfg.experiment.param = cfg_hash
    save_path = (
        Path(cfg.experiment.save_path) /
        cfg.experiment.exp_name /
        (f'param-{cfg.experiment.param}') /
        (f'seed-{cfg.experiment.seed}')
    )
    print(save_path)
    print(save_path)
    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    return save_path


def update_pbar(pbar, stats):
    keys = ['last_bellman_error', 'avg_reward']  # which information to display
    pbar_str = ''
    for k, v in stats.items():
        if k in keys:
            if isinstance(v, float):
                pbar_str += '{key} : {val:.1f}, '.format(key=k, val=v)
            else:
                pbar_str += '{key} : {val} '.format(key=k, val=v)
    pbar.set_description(pbar_str)


def load_transitions(cfg, save_path, env):
    data_loader = init_data_loader(cfg.data_loader)
    # Any way to avoid this if the transition file already exists?
    offline_data_df = data_loader.load_data()
    obs_space_low, obs_space_high = data_loader.get_obs_max_min(
        offline_data_df,
    )

    if cfg.experiment.set_env_obs_space:
        env.observation_space = spaces.Box(
            low=obs_space_low, high=obs_space_high, dtype=np.float32,
        )

    offline_sc = init_state_constructor(cfg.state_constructor, env)
    offline_interaction = init_interaction(cfg.interaction, env, offline_sc)

    reward_func = init_reward_function(cfg.env.reward)
    reward_normalizer = offline_interaction.reward_normalizer
    action_normalizer = offline_interaction.action_normalizer

    # Load transitions
    if (save_path / "offline_transitions.pkl").is_file():
        offline_transitions = data_loader.load_transitions(
            save_path / "offline_transitions.pkl",
        )
    else:
        # In the future, we may just want to pass the entire normalizer in.
        # This assumes you're using a normalizer interaction
        offline_transitions = data_loader.create_transitions(
            offline_data_df, offline_sc, reward_func, action_normalizer,
            reward_normalizer,
        )
        data_loader.save_transitions(
            offline_transitions, save_path / "offline_transitions.pkl",
        )
    train_transitions, test_transitions = data_loader.train_test_split(
        offline_transitions,
    )

    return train_transitions, test_transitions, offline_sc


def get_state_action_dim(env_cfg, env, sc):
    if env_cfg.obs_length == 1:
        obs_shape = (flatdim(env.observation_space), )
    else:
        obs_shape = (env_cfg.obs_length, flatdim(env.observation_space))
    dummy_obs = np.ones(obs_shape)
    state_dim = sc.get_state_dim(dummy_obs)  # gets state_dim dynamically
    action_dim = flatdim(env.action_space)
    return state_dim, action_dim


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    save_path = prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    init_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # load_transitions() will potentially modify the environment by setting
    # obs_space high and low, so we instantiate it first
    env = init_environment(cfg.env)

    do_offline_training = cfg.experiment.offline_steps > 0
    if do_offline_training:
        train_transitions, test_transitions, sc = load_transitions(
            cfg, save_path, env,
        )

    sc = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = get_state_action_dim(cfg.env, env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)
    interaction = init_interaction(cfg.interaction, env, sc, agent)

    # instantiate offline evaluators
    offline_eval_args = {
        'agent': agent
    }
    offline_eval = CompositeEval(cfg.eval, offline_eval_args, offline=True)

    if do_offline_training:
        print('Starting offline training...')
        for transition in train_transitions:
            agent.update_buffer(transition)
        offline_steps = cfg.experiment.offline_steps
        pbar = tqdm(range(offline_steps))
        for _ in pbar:
            agent.update()
            # run all evaluators
            offline_eval.do_eval(**offline_eval_args)
            stats = offline_eval.get_stats()
            update_pbar(pbar, stats)

    # Online Deployment
    # Instantiate online evaluators
    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    ep_reward = 0
    ep_steps = 0
    print('Starting online training...')
    for _ in pbar:
        action = agent.get_action(state)
        transitions, _ = interaction.step(state, action)

        for transition in transitions:
            agent.update_buffer(transition)

        agent.update()

        # logging + evaluation
        # union of the information needed by all evaluators
        online_eval_args = {
            'agent': agent,
            'env': env,
            'transitions': transitions
        }

        # TODO: this is super expensive, maybe we should be doing it every N
        # steps or something...
        online_eval.do_eval(**online_eval_args)
        # stats = online_eval.get_stats()
        # update_pbar(pbar, stats)

        # freezer example
        # fr.freezer.save()

        terminated = transitions[-1][4]
        truncated = transitions[-1][5]
        # print(terminated, truncated)
        reward = transitions[-1][2]
        ep_reward += reward
        ep_steps += 1
        if terminated or truncated:
            state, _ = interaction.reset()
            # print("ep_reward:", ep_reward)
            # print("ep_steps:", ep_steps)
            ep_reward = 0
            ep_steps = 0
        else:
            state = transitions[-1][3]

        # print(state, action)

    online_eval.output(save_path / 'stats.json')
    # need to update make_plots here
    stats = online_eval.get_stats()
    # make_plots(fr.freezer, stats, save_path / 'plots')
    # stats = None

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    return stats


if __name__ == "__main__":
    main()
