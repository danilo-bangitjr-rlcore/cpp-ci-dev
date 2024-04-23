import hydra
import numpy as np
import torch
import random
import queue

from tqdm import tqdm
from omegaconf import DictConfig
from gymnasium.spaces.utils import flatdim

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.interaction.factory import init_interaction
from corerl.utils.evaluator import Evaluator
from corerl.utils.device import init_device
from corerl.utils.multithreading import multithreaded_step
from main import prepare_save_dir, update_pbar
import corerl.utils.freezer as fr


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

    state, info = env.reset()
    state_dim = sc.get_state_dim(state)  # gets state_dim dynamically
    agent = init_agent(cfg.agent, state_dim, action_dim)

    evaluator = Evaluator(cfg.evaluator)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    for _ in pbar:
        action = agent.get_action(state)
        transition = multithreaded_step(agent, interaction, state, action)
        agent.update_buffer(transition)
        evaluator.update(transition)

        state = transition[3]  # next_state is the fourth entry of the transition tuple

        # progress bar logging
        stats = evaluator.get_stats()
        update_pbar(pbar, stats)

    evaluator.output(save_path / 'stats.json')
    return evaluator.get_stats()


if __name__ == "__main__":
    main()
