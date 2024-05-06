import hydra
import numpy as np
import torch
import random

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces.utils import flatdim
from gymnasium import spaces
from pathlib import Path

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.interaction.normalizer_utils import init_action_normalizer, init_reward_normalizer
from corerl.interaction.factory import init_interaction
from corerl.utils.evaluator import Evaluator
from corerl.utils.device import init_device
from corerl.component.data_loaders.factory import init_data_loader
from corerl.environment.reward.factory import init_reward_function
from corerl.utils.plotting import make_plots

import corerl.utils.freezer as fr

"""
Revan: This is an example of how to run the code in the library. 
I expect that each project may need something slightly different than what's here. 
"""


def prepare_save_dir(cfg):
    save_path = (Path(cfg.experiment.save_path) / cfg.experiment.exp_name
                 / ('param-' + str(cfg.experiment.param)) / ('seed-' + str(cfg.experiment.seed)))
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    return save_path

def update_pbar(pbar, stats):
    pbar_str = ''
    for k, v in stats.items():
        if isinstance(v, float):
            pbar_str += '{key} : {val:.1f}, '.format(key=k, val=v)
        else:
            pbar_str += '{key} : {val} '.format(key=k, val=v)
    pbar.set_description(pbar_str)

@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> None:
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
        data_loader = init_data_loader(cfg.data_loader)
        offline_data_df = data_loader.load_data()
        obs_space_low, obs_space_high = data_loader.get_obs_max_min(offline_data_df)
        env.observation_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)

    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)
    action_dim = flatdim(env.action_space)
    state_dim = interaction.get_state_dim()  # gets state_dim dynamically
    agent = init_agent(cfg.agent, state_dim, action_dim)

    if do_offline_training:
        reward_func = init_reward_function(cfg.env.reward)
        reward_normalizer = init_reward_normalizer(cfg.interaction.reward_normalizer)
        action_normalizer = init_action_normalizer(cfg.interaction.action_normalizer, env)
        # Offline Training
        if (save_path / "offline_transitions.pkl").is_file():
            offline_transitions = data_loader.load_transitions(save_path / "offline_transitions.pkl")
        else:
            offline_transitions = data_loader.create_transitions(offline_data_df, sc, reward_func, action_normalizer, reward_normalizer)
            data_loader.save_transitions(offline_transitions, save_path / "offline_transitions.pkl")
        train_transitions, test_transitions = data_loader.train_test_split(offline_transitions)
        for transition in train_transitions:
            agent.update_buffer(transition)
        offline_steps = cfg.experiment.offline_steps
        pbar = tqdm(range(offline_steps))
        for _ in pbar:
            agent.update()

    # Online Deployment
    evaluator = Evaluator(cfg.evaluator)
    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    for _ in pbar:
        action = agent.get_action(state)
        transitions, envinfo = interaction.step(state, action)

        for transition in transitions:
            agent.update_buffer(transition)
            evaluator.update(transition)

        agent.update()
        agent.add_to_freezer()
        state = transitions[-1][0]

        # logging
        stats = evaluator.get_stats()
        update_pbar(pbar, stats)

        # freezer example
        fr.freezer.save()

        # # examples of saving and loading
        # agent.save(save_path / 'agent')
        # agent.load(save_path / 'agent')

    evaluator.output(save_path / 'stats.json')
    make_plots(fr.freezer, save_path / 'plots')

    return evaluator.get_stats()

if __name__ == "__main__":
    main()
