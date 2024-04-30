from typing import Tuple


def avg_reward(params: dict, stats: dict) -> Tuple[bool | None, str]:
    if params['agent'] == 'greedy_ac':
        if params['agent.uniform_proposal'] == False:
            if stats['avg_return '] >= 0.7:
                return True, 'GAC  achieved desired average reward on three tanks'
            else:
                return False, 'GAC failed to achieve desired average reward on three tanks'
        else:
            if stats['avg_return (100)'] >= 0.7:
                return True, 'GAC (uniform prop.) achieved desired average reward on three tanks'
            else:
                return False, 'GAC (uniform prop.) failed to achieve desired average reward on three tanks'

    if params['agent'] == 'greedy_iql':
        if params['agent.uniform_proposal'] == False:
            if stats['avg_return'] == 0.7:
                return True, 'Greedy IQL achieved desired average reward on three tanks'
            else:
                return False, 'Greedy IQL failed to achieve desired average reward on three tanks'

        else:
            if stats['avg_return'] >= 0.7:
                return True, 'Greedy IQL (uniform prop.) achieved desired average reward on three tanks'
            else:
                return False, 'Greedy IQL (uniform prop.) failed to achieve desired average reward on three tanks'
    else:
        return None, 'n/a'

def avg_reward_last_iterations(params: dict, stats: dict) -> Tuple[bool | None, str]:
    if params['agent'] == 'greedy_ac':
        if params['agent.uniform_proposal'] == False:
            if stats['avg_return (100)'] == 1.0:
                return True, 'GAC  achieved desired average reward on three tanks'
            else:
                return False, 'GAC failed to achieve desired average reward on three tanks'

        else:
            if stats['avg_return (100)'] >= 0.7:
                return True, 'GAC (uniform prop.) achieved desired average reward on three tanks'
            else:
                return False, 'GAC (uniform prop.) failed to achieve desired average reward on three tanks'

    if params['agent'] == 'greedy_iql':
        if params['agent.uniform_proposal'] == False:
            if stats['avg_return (100)'] == 1.0:
                return True, 'Greedy IQL achieved desired average reward on three tanks'
            else:
                return False, 'Greedy IQL failed to achieve desired average reward on three tanks'

        else:
            if stats['avg_return (100)'] >= 0.7:
                return True, 'Greedy IQL (uniform prop.) achieved desired average reward on three tanks'
            else:
                return False, 'Greedy IQL (uniform prop.) failed to achieve desired average reward on three tanks'
    else:
        return None, 'n/a'


SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac'],
        'experiment.max_steps': [1000],
        'env': ['three_tanks'],
        'experiment.seed': [0],
        'experiment.exp_name': ['test_three_tanks'],
        'interaction/reward_normalizer': ['clip'],
        'interaction.reward_normalizer.clip_min': [-2]
    },

    'conditional': {
        'agent.rho': lambda d: [0.1],
        'agent.expectile': lambda d: [0.8] if d['agent'] == 'greedy_iql' else None,
        'agent.num_samples': lambda d: [300],
        'agent.uniform_proposal': lambda d: [True]
    },

    'tests': [avg_reward, avg_reward_last_iterations]
}
