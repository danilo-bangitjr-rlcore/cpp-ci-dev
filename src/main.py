from pathlib import Path

import jax
from coreenv.factory import init_env
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Identity, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from tqdm import tqdm

import utils.gym as gym_u
from agent.gac import GreedyAC, GreedyACConfig
from interaction.env_wrapper import EnvWrapper
from metrics.actor_critic import ac_eval


def main():
    seed = 0

    collector = Collector(
        tmp_file='./test.db',
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
            'action_0': Pipe(
                MovingAverage(0.99),
                Subsample(100),
            ),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Identity(),
        experiment_id=seed,
        low_watermark=1,
    )

    env = init_env('Saturation-v0')
    # env = gym.make("MountainCarContinuous-v0")

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
        GreedyACConfig(),
        seed=seed,
        state_dim=wrapper_env.get_state_dim(),
        action_dim=1,
        collector=collector,
    )
    rng = jax.random.PRNGKey(0)

    episodes = 2
    for _ in range(episodes):
        state, _ = wrapper_env.reset()
        episode_reward = 0.0
        steps = 0
        pbar = tqdm()
        while True:
            collector.next_frame()
            rng, step_key = jax.random.split(rng)
            ac_eval(collector, agent, state)
            action = agent.get_actions(state)

            next_state, reward, terminated, truncated, info, transitions = wrapper_env.step(action)
            for t in transitions:
                agent.update_buffer(t)

            agent.update()
            for i, x in enumerate(next_state):
                collector.collect(f'state_{i}', float(x))

            for i, a in enumerate(action):
                collector.collect(f'action_{i}', float(a))

            collector.collect('reward', reward)
            episode_reward += reward
            steps += 1
            pbar.update(1)

            if terminated or truncated:
                break

            state = next_state

    env.close()

    save_path = Path('results/test/results.db')
    save_path.parent.mkdir(exist_ok=True, parents=True)
    # TODO: get the hyperparam values from the configs
    hyperparams = { 'stepsize': 0.0001 }

    # add the hyperparameters to the results database
    attach_metadata(save_path, collector.get_current_experiment_id(), hyperparams)
    collector.merge(str(save_path))
    collector.close()

if __name__ == "__main__":
    main()
