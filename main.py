import hydra
import numpy as np
import torch
import random
import time
import hashlib
import copy
import pickle as pkl

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
from corerl.data_loaders.utils import make_anytime_transitions, train_test_split
import corerl.utils.freezer as fr


def prepare_save_dir(cfg):
    if cfg.experiment.param_from_hash:
        cfg_copy = copy.deepcopy(cfg)
        del cfg_copy.experiment.seed
        cfg_hash = hashlib.sha1(str(cfg_copy).encode("utf-8")).hexdigest()
        print("Creating experiment param from hash:", cfg_hash)
        cfg.experiment.param = cfg_hash
    save_path = (
            Path(cfg.experiment.save_path) /
            cfg.experiment.exp_name /
            (f'param-{cfg.experiment.param}') /
            (f'seed-{cfg.experiment.seed}')
    )
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


def check_exists(save_path):
    if save_path.exists():
        with open(save_path, 'rb') as f:
            return pkl.load(f)
    else:
        return None


def load_or_create(root, cfgs, prefix, create_func, args):
    cfg_str = ''
    for cfg in cfgs:
        cfg_copy = OmegaConf.to_container(copy.deepcopy(cfg), resolve=True)
        cfg_str += str(cfg_copy)

    cfg_hash = hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()
    save_path = root / cfg_hash / f"{prefix}-{cfg_hash}.pkl"
    obj = check_exists(save_path)

    if obj is None:
        print(f"Generating {prefix}...")
        obj = create_func(*args)  # loads the entire dataset

        save_path = root / cfg_hash
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

        with open(save_path / f"{prefix}-{cfg_hash}.pkl", 'wb') as f:
            pkl.dump(obj, f)

        print(f"Saved {prefix} to {save_path}.")
    else:
        print(f"Loaded {prefix} from {save_path}.")

    return obj


def load_offline_data_from_csv(cfg):
    env = init_environment(cfg.env)
    dl = init_data_loader(cfg.data_loader)

    output_path = Path(cfg.offline_data.output_path)

    create_df = lambda dl_, filenames: dl_.load_data(filenames)
    all_data_df = load_or_create(output_path, [cfg.data_loader],
                                 'all_data_df', create_df, [dl, dl.all_filenames])
    train_data_df = load_or_create(output_path, [cfg.data_loader],
                                   'train_data_df', create_df, [dl, dl.train_filenames])
    test_data_df = load_or_create(output_path, [cfg.data_loader],
                                  'test_data_df', create_df, [dl, dl.test_filenames])

    assert not all_data_df.isnull().values.any()
    assert not train_data_df.isnull().values.any()
    assert not test_data_df.isnull().values.any()

    create_bounds = lambda dl_, df: dl.get_obs_max_min(df)
    obs_bounds = load_or_create(output_path, [cfg.data_loader],
                                'obs_bounds', create_bounds, [dl, all_data_df])

    env.observation_space = spaces.Box(low=obs_bounds[0], high=obs_bounds[1], dtype=np.float32)
    sc = init_state_constructor(cfg.state_constructor, env)

    reward_func = init_reward_function(cfg.env.reward)
    create_obs_transitions = lambda dl_, r_func, df: dl_.create_obs_transitions(df, reward_func)
    train_obs_transitions = load_or_create(output_path, [cfg.data_loader],
                                           'train_obs_transitions', create_obs_transitions,
                                           [dl, reward_func, train_data_df])

    if test_data_df is not None:
        test_obs_transitions = load_or_create(output_path, [cfg.data_loader],
                                              'test_obs_transitions', create_obs_transitions,
                                              [dl, reward_func, test_data_df])
    else:
        test_obs_transitions = None

    interaction = init_interaction(cfg.interaction, env, sc)
    create_transitions = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_transitions(
        obs_transitions,
        interaction_,
        sc_warmup=cfg.state_constructor.warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )
    train_transitions, _ = load_or_create(output_path,
                                            [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                            'train_transitions', create_transitions,
                                            [train_obs_transitions, interaction, cfg.state_constructor.warmup, False])

    if test_obs_transitions is not None:
        test_transitions, test_scs = load_or_create(output_path,
                                                    [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                                    'test_transitions', create_transitions,
                                                    [test_obs_transitions, interaction, cfg.state_constructor.warmup,
                                                     True])
    else:
        test_transitions = None
        test_scs = None

    return env, sc, interaction, train_transitions, test_transitions, test_scs


def load_offline_data_from_transitions(cfg):
    output_path = Path(cfg.offline_data.output_path)

    # We assume that transitions have been created with make_offline_transitions.py
    nothing_fn = lambda *args: None

    train_obs_transitions = load_or_create(output_path,
                                           [cfg.env, cfg.state_constructor, cfg.interaction, cfg.agent],
                                           'obs_transitions', nothing_fn,
                                           [])


    # TODO make this actually return something
    test_obs_transitions = None

    env = init_environment(cfg.env)
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)

    create_transitions = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_transitions(
        obs_transitions,
        interaction_,
        sc_warmup=cfg.state_constructor.warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )
    train_transitions, _ = load_or_create(output_path,
                                            [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                            'train_transitions', create_transitions,
                                            [train_obs_transitions, interaction, cfg.state_constructor.warmup, False])

    if test_obs_transitions is not None:
        test_transitions, test_scs = load_or_create(output_path,
                                                    [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                                    'test_transitions', create_transitions,
                                                    [test_obs_transitions, interaction, cfg.state_constructor.warmup,
                                                     True])
    else:
        test_transitions = None
        test_scs = None

    return env, sc, interaction, train_transitions, test_transitions, test_scs

def get_state_action_dim(env, sc):
    obs_shape = (flatdim(env.observation_space),)
    dummy_obs = np.ones(obs_shape)
    action_shape = (flatdim(env.action_space),)
    dummy_action = np.ones(action_shape)
    state_dim = sc.get_state_dim(dummy_obs, dummy_action)  # gets state_dim dynamically
    action_dim = flatdim(env.action_space)
    return state_dim, action_dim


def offline_training(cfg, agent, train_transitions, test_transitions):
    print('Starting offline training...')
    offline_eval_args = {
        'agent': agent
    }
    offline_eval = CompositeEval(cfg.eval, offline_eval_args, offline=True)

    if test_transitions is None:
        split = train_test_split(train_transitions, train_split=cfg.experiment.train_split)
        train_transitions, test_transitions = split[0][0], split[0][1]

    for transition in train_transitions:
        agent.update_buffer(transition)

    offline_steps = cfg.experiment.offline_steps
    pbar = tqdm(range(offline_steps))
    for _ in pbar:
        agent.update()
        offline_eval.do_eval(**offline_eval_args)  # run all evaluators
        stats = offline_eval.get_stats()
        update_pbar(pbar, stats)

    return offline_eval


def online_deployment(cfg, agent, interaction, env):
    # Online Deployment
    # Instantiate online evaluators
    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    print('Starting online training...')
    for _ in pbar:
        action = agent.get_action(state)
        transitions, _ = interaction.step(action)

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

        online_eval.do_eval(**online_eval_args)
        stats = online_eval.get_stats()
        update_pbar(pbar, stats)

        terminated = transitions[-1].terminated
        truncated = transitions[-1].truncate
        if terminated or truncated:
            state, _ = interaction.reset()
        else:
            state = transitions[-1].boot_state

    return online_eval


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

    do_offline_training = cfg.experiment.offline_steps > 0

    if do_offline_training:
        print('Loading offline transitions...')
        if cfg.offline_data.load_from == 'csv':
            env, sc, interaction, train_transitions, test_transitions, _ = load_offline_data_from_csv(cfg)
        elif cfg.offline_data.load_from == 'transition':
            env, sc, interaction, train_transitions, test_transitions, test_scs = load_offline_data_from_transitions(cfg)
    else:
        env = init_environment(cfg.env)
        sc = init_state_constructor(cfg.state_constructor, env)
        interaction = init_interaction(cfg.interaction, env, sc)

    state_dim, action_dim = get_state_action_dim(env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)

    if do_offline_training:
        offline_eval = offline_training(cfg, agent, train_transitions, test_transitions)

    online_eval = online_deployment(cfg, agent, interaction, env)
    online_eval.output(save_path / 'stats.json')\

    env.plot()

    # need to update make_plots here
    stats = online_eval.get_stats()
    make_plots(fr.freezer, stats, save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    return stats


if __name__ == "__main__":
    main()
