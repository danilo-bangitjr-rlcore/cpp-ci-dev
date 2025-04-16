import argparse
import ast
from pathlib import Path
from typing import Any

from coreenv.factory import init_env
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Identity, Subsample, Window
from tqdm import tqdm

import utils.gym as gym_u
from agent.gac import GreedyAC, GreedyACConfig
from config.experiment import ExperimentConfig, get_next_id
from interaction.env_wrapper import EnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)

args, override_args = parser.parse_known_args()

def safe_cast(value: Any):
    if not isinstance(value, str):
        return value

    try:
        # Check if it's a valid Python literal (number, list, dict, etc.)
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If not a Python literal, it might be a regular string
        # or a string representation of True/False
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            return value  # Keep as string


def process_overrides(override_args: list) -> list[tuple]:
    """
    Process the overrides from the command line arguments.
    """
    override_list = []
    for override in override_args:
        key, value = override.split('=')
        value = safe_cast(value)

        keys = key.split('.')
        override_list.append((keys, value))
    return override_list


def main():
    overrides = process_overrides(override_args)
    cfg = ExperimentConfig.load(args.exp, overrides)

    save_path = Path('results/test/results.db')
    save_path.parent.mkdir(exist_ok=True, parents=True)

    exp_id = get_next_id(save_path)

    collector = Collector(
        tmp_file=str(save_path),
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'reward': Window(100),
            'critic_loss': Subsample(100),
            'actor_mu_0': Subsample(100),
            'actor_sigma_0': Subsample(100),
            'actor_loss': Subsample(100),
            'actor_grad_norm': Subsample(100),
            'proposal_loss': Subsample(100),
            'proposal_grad_norm': Subsample(100),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Identity(),
        experiment_id=exp_id,
        low_watermark=1,
    )

    if cfg.env['name'] == 'WindyRoom-v0':
       env_args = {'no_zones': False}
    else:
        env_args = {}

    if cfg.env['name'] == 'DistractionWorld-v0':
        trace_values = (0.,)
    else:
        trace_values = (0., 0.75, 0.9, 0.95)

    env = init_env(cfg.env['name'], env_args)

    obs_bounds = gym_u.space_bounds(env.observation_space)
    act_bounds = gym_u.space_bounds(env.action_space)
    wrapper_env = EnvWrapper(
        env=env,
        action_space_info={
            'low': act_bounds[0],
            'high': act_bounds[1],
        },
        observation_space_info={
            'low': obs_bounds[0],
            'high': obs_bounds[1],
        },
        min_n_step=1,
        max_n_step=1,
        gamma=0.99,
        trace_values=trace_values,
    )

    agent = GreedyAC(
        GreedyACConfig(**cfg.agent),
        seed=args.seed,
        state_dim=wrapper_env.get_state_dim(),
        action_dim=1,
        collector=collector,
    )

    state, _ = wrapper_env.reset()
    episode_reward = 0.0
    steps = 0
    # +1 to ensure we don't reject any metrics that are subsampling
    # every 100 steps
    for _ in tqdm(range(cfg.max_steps + 1)):
        collector.next_frame()
        # ac_eval(collector, agent, state)
        action = agent.get_actions(state)

        next_state, reward, terminated, truncated, info, transitions = wrapper_env.step(action)
        for t in transitions:
            agent.update_buffer(t)

        agent.update()
        collector.collect('reward', reward)
        episode_reward += reward
        steps += 1

        if terminated or truncated:
            collector.collect('return', episode_reward)
            collector.collect('num_steps', steps)

            steps = 0
            episode_reward = 0

            state, _ = wrapper_env.reset()

        else:
            state = next_state

    env.close()

    # add the hyperparameters to the results database
    hyperparams = cfg.flatten()
    hyperparams['seed'] = args.seed
    attach_metadata(save_path, collector.get_current_experiment_id(), hyperparams)
    collector.reset()
    collector.close()

if __name__ == "__main__":
    main()
