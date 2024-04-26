import hydra
import numpy as np
import torch
import random

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces.utils import flatdim
from pathlib import Path

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.interaction.factory import init_interaction
from corerl.utils.evaluator import Evaluator
from corerl.utils.device import init_device

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
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)
    action_dim = flatdim(env.action_space)

    state_dim = interaction.get_state_dim()  # gets state_dim dynamically
    agent = init_agent(cfg.agent, state_dim, action_dim)
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
        state = transitions[-1][0]

        # logging
        stats = evaluator.get_stats()
        update_pbar(pbar, stats)

        # freezer example
        fr.freezer['transitions'] = transitions[-1]
        fr.freezer.save()
        fr.freezer.increment()
        fr.freezer.clear()  # Optionally clearing the log

        # examples of saving and loading
        agent.save(save_path / 'agent')
        agent.load(save_path / 'agent')

    evaluator.output(save_path / 'stats.json')
    return evaluator.get_stats()

if __name__ == "__main__":
    main()
