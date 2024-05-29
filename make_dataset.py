import hydra
import numpy as np
import torch
import random

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces.utils import flatdim
from pathlib import Path
from copy import deepcopy

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.eval.composite_eval import CompositeEval
from corerl.interaction.factory import init_interaction
from corerl.utils.device import init_device
from corerl.utils.plotting import make_plots
from corerl.calibration_models.utils import Trajectory


def prepare_save_dir(cfg):
    save_path = (Path(cfg.experiment.save_path) / cfg.experiment.exp_name
                 / ('param-' + str(cfg.experiment.param)) / ('seed-' + str(cfg.experiment.seed)))
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


def get_state_action_dim(env, sc):
    obs_shape = (flatdim(env.observation_space),)
    dummy_obs = np.ones(obs_shape)
    action_shape = (flatdim(env.action_space),)
    dummy_action = np.ones(action_shape)
    state_dim = sc.get_state_dim(dummy_obs, dummy_action)  # gets state_dim dynamically
    action_dim = flatdim(env.action_space)
    return state_dim, action_dim


@hydra.main(version_base=None, config_name='reseau', config_path="config/")
def main(cfg: DictConfig) -> dict:
    save_path = prepare_save_dir(cfg)
    init_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = init_environment(cfg.env)

    # we must instantiate the sc after we set env.observation_space since normalization depends on these values
    sc = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = get_state_action_dim(env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)
    interaction = init_interaction(cfg.interaction, env, sc, agent, action_dim)

    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    print('Starting online training...')

    traj = Trajectory(deepcopy(sc))
    for itr in pbar:
        action = agent.get_action(state)
        transitions, _ = interaction.step(action)

        for transition in transitions:
            agent.update_buffer(transition)

        agent.update()

        # logging + evaluation
        online_eval_args = {  # union of the information needed by all evaluators
            'agent': agent,
            'transitions': transitions
        }
        online_eval.do_eval(**online_eval_args)
        stats = online_eval.get_stats()
        update_pbar(pbar, stats)

        # add to trajectories
        for transition in transitions:
            traj.add_transition(transition)


        # check if terminated, and get the next state from the list of transitions
        terminated = transitions[-1].terminated
        truncated = transitions[-1].truncate
        if terminated or truncated:
            state, _ = interaction.reset()
        else:
            state = transitions[-1].next_state

    env.plot()
    online_eval.output(save_path / 'stats.json')
    # need to update make_plots here
    stats = online_eval.get_stats()

    return stats


if __name__ == "__main__":
    main()
