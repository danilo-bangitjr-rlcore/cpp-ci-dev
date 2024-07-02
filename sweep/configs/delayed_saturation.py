SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac'],
        'interaction.steps_per_decision': [1, 5, 10, 20],
        'interaction.n_step': [0, 1],
        'env.trace_val': [0, 0.5, 0.75, 0.9, 0.99],
        'experiment.seed': list(range(10))
    },

    'conditional': {
        # 'interaction.': lambda d: [0.1, 0.5, 0.8] if d['agent'] == 'iql' else None,
    }
}
