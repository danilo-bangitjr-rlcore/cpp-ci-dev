import argparse
import ast
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from lib_agent.actor.percentile_actor import State
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import config_from_dict
from lib_utils.named_array import NamedArray
from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Identity, Subsample, Window
from rl_env.factory import EnvConfig, init_env
from tqdm import tqdm

import utils.gym as gym_u
from agent.gac import GreedyACConfig
from config.experiment import ExperimentConfig, get_next_id
from interaction.env_wrapper import EnvWrapper
from interaction.goal_constructor import Goal, GoalConstructor, RewardConfig, TagConfig
from interaction.transition_creator import TransitionCreator
from src.agent.factory import get_agent
from utils.action_bounds import DeltaActionBoundsComputer
from utils.plotting import plot_learning_curve

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('--plot', action='store_true', help='Plot learning curve after training')
parser.add_argument('--save-path', type=str, default='results/test', help='Directory to save results')

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
        if value.lower() == 'false':
            return False
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


def flatten_config(cfg: dict[str, Any]):
    flattened = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            nested = flatten_config(value)
            for nested_key, nested_value in nested.items():
                flattened[f"{key}.{nested_key}"] = nested_value
        elif isinstance(value, list):
            flattened[key] = str(value)  # Convert list to string representation
        else:
            flattened[key] = value
    return flattened


def main():
    overrides = process_overrides(override_args)
    cfg = ExperimentConfig.load(args.exp, overrides)

    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)

    db_path = save_dir / 'results.db'

    hyperparams = flatten_config(cfg.flatten())
    hyperparams['seed'] = args.seed
    exp_id = get_next_id(db_path, hyperparams)

    collector = Collector(
        tmp_file=str(db_path),
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'reward': Window(25),
            'critic_loss': Subsample(100),
            'actor_mu_0': Subsample(100),
            'actor_sigma_0': Subsample(100),
            'actor_loss': Subsample(100),
            'actor_grad_norm': Subsample(100),
            'proposal_loss': Subsample(100),
            'proposal_grad_norm': Subsample(100),
        },
        default=Identity(),
        experiment_id=exp_id,
        low_watermark=1,
    )

    if cfg.env['name'] == 'WindyRoom-v0':
        cfg.env['no_zones'] = False

    trace_values = (0., 0.75, 0.9, 0.95)
    if cfg.env['name'] == 'DistractionWorld-v0':
        trace_values = (0.,)

    env_cfg = config_from_dict(EnvConfig, cfg.env)  # type: ignore
    assert not isinstance(env_cfg, ConfigValidationErrors)
    env = init_env(env_cfg)

    obs_bounds = gym_u.space_bounds(env.observation_space)
    act_bounds = gym_u.space_bounds(env.action_space)

    goal_constructor = None
    if 'reward' in cfg.pipeline:
        tag_configs = [
            TagConfig(
                name=f'tag-{i}',
                operating_range=(obs_bounds[0][i], obs_bounds[1][i]),
            )
            for i in range(len(obs_bounds[0]))
        ]

        priorities = [
            Goal(tag=goal['tag'], op=goal['op'], thresh=goal['thresh'])
            for goal in cfg.pipeline['reward']['priorities']
        ]

        reward_config = RewardConfig(
            priorities=priorities,
        )
        goal_constructor = GoalConstructor(reward_config, tag_configs)

    wrapper_env = EnvWrapper(
        env=env,
        collector=collector,
        action_space_info={
            'low': act_bounds[0],
            'high': act_bounds[1],
        },
        observation_space_info={
            'low': obs_bounds[0],
            'high': obs_bounds[1],
        },
        trace_values=trace_values,
        goal_constructor=goal_constructor,
    )
    tc = TransitionCreator(n_step=1, gamma=0.99)

    agent = get_agent(
        GreedyACConfig(**cfg.agent),
        seed=args.seed,
        state_dim=wrapper_env.get_state_dim(),
        action_dim=len(act_bounds[0]),
        collector=collector,
    )

    # Initialize action bounds computer
    bounds_computer = DeltaActionBoundsComputer(
        config=cfg.action_bounds,
        action_dim=len(act_bounds[0]),
    )

    state_features, _ = wrapper_env.reset()
    reward: float | None = None
    done = False
    episode_reward = 0.0
    steps = 0
    # +1 to ensure we don't reject any metrics that are subsampling
    # every 100 steps

    # Initialize last_action to middle of static bounds
    last_action = (bounds_computer.static_lo + bounds_computer.static_hi) / 2
    dp = True

    for _ in tqdm(range(cfg.max_steps + 1)):
        dp = steps % cfg.steps_per_decision == 0
        collector.next_frame()

        # Compute action bounds based on previous action
        a_lo, a_hi = bounds_computer.compute(last_action)

        state = State(
            features=NamedArray.unnamed(jnp.array(state_features)),
            a_lo=jnp.array(a_lo),
            a_hi=jnp.array(a_hi),
            dp=jnp.array([dp]),
            last_a=jnp.array(last_action),
        )
        action = agent.get_actions(state)

        transitions = tc(
            state_features, a_lo, a_hi, np.array(action), reward, done, dp)

        next_state_features, reward, terminated, truncated, _ = wrapper_env.step(action)
        for t in transitions:
            agent.update_buffer(t)

        agent.update()
        collector.collect('reward', reward)
        episode_reward += reward
        steps += 1

        done = terminated or truncated
        if terminated or truncated:
            collector.collect('return', episode_reward)
            collector.collect('num_steps', steps)

            steps = 0
            episode_reward = 0
            reward = None

            state_features, _ = wrapper_env.reset()
            last_action = (bounds_computer.static_lo + bounds_computer.static_hi) / 2
            tc.flush()

        else:
            state_features = next_state_features
            last_action = np.array(action)

    env.close()

    collector.reset()
    collector.close()

    if args.plot:
        plot_learning_curve(save_dir, db_path, exp_id, args.seed)


if __name__ == "__main__":
    main()
