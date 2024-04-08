import hydra
import os
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces.utils import flatdim
from pathlib import Path

from datetime import datetime


from root.agent.factory import init_agent
from root.environment.factory import init_environment
from root.state_constructor.factory import init_state_constructor
from root.interaction.factory import init_interaction
from root.step_log import init_step_log
import root.step_log as log

"""
Revan: This is an example of how to run the code in the library. 
I expect that each project may need something slightly different than what's here. 
"""
def prepare_save_dir(cfg):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    dt_strings = dt_string.split("_")
    # not sure what the exact hierarchy should be
    save_path = (Path(cfg.experiment.save_path) / cfg.experiment.exp_name
                 / dt_strings[0] / dt_strings[1] / cfg.experiment.exp_info / str(cfg.experiment.seed))
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path/"config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    return save_path

@hydra.main(version_base=None, config_name='config', config_path="config")
def main(cfg: DictConfig) -> None:

    save_path = prepare_save_dir(cfg)
    init_step_log(save_path/'logs')



    env = init_environment(cfg.env)
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)
    action_dim = flatdim(env.action_space)

    state, info = env.reset()
    state_dim = sc.get_state_dim(state)  # gets state_dim dynamically
    agent = init_agent(cfg.agent, state_dim, action_dim)

    max_steps = cfg.experiment.max_steps
    for step in tqdm(range(max_steps)):
        action = agent.get_action(state)
        next_state, reward, terminated, truncate, env_info = interaction.step(action)
        transition = (state, action, reward, next_state, terminated, truncate)

        agent.update_buffer(transition)
        agent.update()
        state = next_state

        # logging example
        log.LOG['transition'] = transition
        log.LOG.save()
        log.LOG.increment()
        log.LOG.clear()  # Optionally clearing the log



        # examples of saving and loading
        agent.save(save_path / 'agent')
        agent.load(save_path / 'agent')



if __name__ == "__main__":
    main()
