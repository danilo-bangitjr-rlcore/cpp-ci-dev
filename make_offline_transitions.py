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
from main import *


def update_pbar(pbar):
    pbar.set_description('')


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
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)
    state_dim, action_dim = get_state_action_dim(env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    print('Starting online training...')
    transitions = []
    for _ in pbar:
        action = agent.get_action(state)
        step_transitions, _ = interaction.step(action)

        for transition in step_transitions:
            agent.update_buffer(transition)
            transitions.append(transition)

        agent.update()
        update_pbar(pbar)

        terminated = step_transitions[-1].terminated
        truncated = step_transitions[-1].truncate
        if terminated or truncated:
            state, _ = interaction.reset()
        else:
            state = step_transitions[-1].boot_state

    # save all the transitions
    output_path = Path(cfg.offline_data.output_path)

    def create_transitions(transitions_):
        return transitions_

    load_or_create(output_path,
                   [cfg.env, cfg.state_constructor, cfg.interaction, cfg.agent],
                   'transitions', create_transitions,
                   [transitions])

    def create_obs_transitions(transitions_):
        obs_transitions = []
        for t in transitions_:
            obs_transitions.append(t.to_obs_transition())

        return obs_transitions

    load_or_create(output_path,
                   [cfg.env, cfg.state_constructor, cfg.interaction, cfg.agent],
                   'obs_transitions', create_obs_transitions,
                   [transitions])


if __name__ == "__main__":
    main()
