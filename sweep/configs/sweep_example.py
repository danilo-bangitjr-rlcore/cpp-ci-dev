import numpy as np

SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac', 'iql'],
        'experiment.max_steps': [10]
    },
    'conditional': {
        # syntax: if agent = iql, then 'agent.expectile' = range(0, 10
        'agent.expectile': ['agent', 'iql',  [0.1, 0.5, 0.8]],
        'agent.rho': ['agent', 'greedy_ac', [0.1, 0.2, 0.4]]
    }
}
