import numpy as np

SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac'],
        'experiment.max_steps': [1000],
        'env': ['reactor_env'],
        'experiment.exp_name': ['test_reactor_env']
    },

    'conditional': {
        'agent.rho': lambda d: [0.2] if d['agent'] == 'greedy_ac' else None,
    }
}
