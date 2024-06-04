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


def load_offline_data(cfg, save_path, interaction, data_loader, offline_data_df):
    reward_func = init_reward_function(cfg.env.reward)

    # Load observation_transitions
    obs_transition_filename = "obs_transitions.pkl"
    if (save_path / obs_transition_filename).is_file():
        obs_transitions = data_loader.load(save_path / obs_transition_filename)
    else:
        print("Loading offline transitions...")
        # offline_data = data_loader.create_trajectories(offline_data_df,
        #                                                interaction.state_constructor,
        #                                                reward_func,
        #                                                interaction)
        # offline_data = data_loader.create_transitions(offline_data_df,
        #                                                interaction.state_constructor,
        #                                                reward_func,
        #                                                interaction)
        obs_transitions = data_loader.create_obs_transitions(offline_data_df, reward_func)

        print("Saving data...")
        data_loader.save(obs_transitions, save_path / obs_transition_filename)

    # next, process these observations


    from corerl.data_loaders.utils import make_transitions, make_anytime_transitions
    transitions = make_anytime_transitions(obs_transitions, interaction, sc_warmup=10)
    return transitions


def get_state_action_dim(env, sc):
    obs_shape = (flatdim(env.observation_space),)
    dummy_obs = np.ones(obs_shape)
    action_shape = (flatdim(env.action_space),)
    dummy_action = np.ones(action_shape)
    state_dim = sc.get_state_dim(dummy_obs, dummy_action)  # gets state_dim dynamically
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

    env = init_environment(cfg.env)
    do_offline_training = cfg.experiment.offline_steps > 0

    if do_offline_training:
        # first load transitions if we are doing offline training
        data_loader = init_data_loader(cfg.data_loader)
        offline_data_df = data_loader.load_data()

        if cfg.experiment.set_env_obs_space:
            obs_space_low, obs_space_high = data_loader.get_obs_max_min(offline_data_df)
            env.observation_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)

    # we must instantiate the sc after we set env.observation_space since normalization depends on these values
    sc = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = get_state_action_dim(env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)
    interaction = init_interaction(cfg.interaction, env, sc, agent, action_dim)

    if do_offline_training:
        print('Loading offline transitions...')
        offline_data = load_offline_data(cfg, save_path, interaction, data_loader, offline_data_df)
        # instantiate offline evaluators
        offline_eval_args = {
            'agent': agent
        }
        offline_eval = CompositeEval(cfg.eval, offline_eval_args, offline=True)

        # train calibration model
        offline_data['interaction'] = interaction
        # cm = init_calibration_model(cfg.calibration_model, offline_data)
        # cm.train()

        print('Starting offline training...')
        train_transitions, test_transitions = offline_data['train_transitions'], offline_data['test_transitions']
        for transition in train_transitions:
            agent.update_buffer(transition)
        offline_steps = cfg.experiment.offline_steps
        pbar = tqdm(range(offline_steps))
        for _ in pbar:
            agent.update()
            offline_eval.do_eval(**offline_eval_args)  # run all evaluators
            stats = offline_eval.get_stats()
            update_pbar(pbar, stats)

        # rollout in calibration models
        # cm.do_test_rollouts(agent, rollout_len=100, num_rollouts=1)

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

    env.plot()

    online_eval.output(save_path / 'stats.json')
    # need to update make_plots here
    stats = online_eval.get_stats()
    make_plots(fr.freezer, stats, save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    return stats


if __name__ == "__main__":
    main()
