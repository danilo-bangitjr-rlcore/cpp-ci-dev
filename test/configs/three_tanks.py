import numpy as np

SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac'],
        'experiment.max_steps': [1000],
        'env': ['three_tanks'],
        'experiment.exp_name': ['test_three_tanks']
    },

    'conditional': {
        'agent.rho': lambda d: [0.2] if d['agent'] == 'greedy_ac' else None,
    }
}
