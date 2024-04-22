SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac', 'iql'],
        'experiment.max_steps': [2]
    },

    'conditional': {
        # syntax: if agent == iql, then 'agent.expectile' = [0.1, 0.5, 0.8]
        # d stands for SWEEP_PARAMS['independent']
        'agent.expectile': lambda d: [0.1, 0.5, 0.8] if d['agent'] == 'iql' else None,
        'agent.rho': lambda d: [0.1, 0.3] if d['agent'] == 'greedy_ac' else None,
    }
}
