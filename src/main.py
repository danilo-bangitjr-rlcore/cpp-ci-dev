import argparse
from pathlib import Path

from coreenv.factory import init_env
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Identity, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from tqdm import tqdm

import utils.gym as gym_u
from agent.gac import GreedyAC, GreedyACConfig
from config.experiment import ExperimentConfig, get_next_id
from interaction.env_wrapper import EnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)

args = parser.parse_args()


def main():
    cfg = ExperimentConfig.load(args.exp)

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
            'reward': Pipe(
                MovingAverage(0.99),
                Subsample(100),
            ),
            'critic_loss': Subsample(100),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Identity(),
        experiment_id=exp_id,
        low_watermark=1,
    )

    env = init_env(cfg.env['name'], cfg.env)

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
