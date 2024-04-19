from typing import Tuple


def avg_reward(params: dict, stats: dict) -> Tuple[bool | None, str]:
    if params['agent'] == 'greedy_ac':
        if stats['avg_reward'] > 0.75:
            return True, 'GAC achieved desired average reward on three tanks'
        else:
            return False, 'GAC failed to achieve desired average reward on three tanks'
    else:
        return None, 'n/a'


def avg_reward_last_iterations(params: dict, stats: dict) -> Tuple[bool | None, str]:
    if params['agent'] == 'greedy_ac':
        if stats['avg_return (100)'] == 1.0:
            return True, 'GAC achieved desired average reward on three tanks'
        else:
            return False, 'GAC failed to achieve desired average reward on three tanks'
    else:
        return None, 'n/a'


SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac'],
        'experiment.max_steps': [100],
        'env': ['three_tanks'],
        'experiment.exp_name': ['test_three_tanks'],
        'interaction/reward_normalizer': ['clip'],
        'interaction.reward_normalizer.clip_min': [-2]
    },

    'conditional': {
        'agent.rho': lambda d: [0.2] if d['agent'] == 'greedy_ac' else None,
    },

    'tests': [avg_reward, avg_reward_last_iterations]
}
