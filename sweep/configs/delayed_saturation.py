SWEEP_PARAMS = {
    'agent': ['greedy_ac'],
    'interaction.steps_per_decision': [1, 2, 5, 10],
    'env.trace_val': [0, 0.5, 0.75, 0.9, 0.95],
    'experiment.seed': list(range(10)),
    'interaction.only_dp_transitions': [True, False],
    'interaction.n_step': lambda d: [0, d['interaction.steps_per_decision']] if d['interaction.steps_per_decision'] > 1 and not d['interaction.only_dp_transitions'] else None,
    'state_constructor': lambda d: ['multi_trace'] if d['interaction.only_dp_transitions'] else ['anytime_multi_trace'],
    'state_constructor.representation': lambda d: ['countdown', 'one_hot', 'thermometer'] if d['state_constructor'] == 'anytime_multi_trace' else None,
    'state_constructor.use_indicator': lambda d: [False, True] if d['state_constructor'] == 'anytime_multi_trace' else None,
}
