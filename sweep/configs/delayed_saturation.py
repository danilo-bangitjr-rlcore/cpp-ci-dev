SWEEP_PARAMS = {
    'agent': ['greedy_ac'],
    'interaction.steps_per_decision': [1, 2, 5, 10],
    'env.trace_val': [0, 0.5, 0.75, 0.9, 0.95],
    'experiment.seed': list(range(10)),
    'interaction.only_dp_transitions': [True, False],
    'interaction.n_step': lambda d: [0, d['interaction.steps_per_decision']] if d['interaction.steps_per_decision'] > 1 else None,

    # 'independent': {
    #     'agent': ['greedy_ac'],
    #     'interaction.steps_per_decision': [1, 2, 5, 10],
    #     'env.trace_val': [0, 0.5, 0.75, 0.9, 0.95],
    #     'experiment.seed': list(range(10)),
    #     'interaction.only_dp_transitions': [True, False],
    # },
    #
    # 'conditional': {
    #     'interaction.n_step': lambda d: [0, d['interaction.steps_per_decision']] if d['interaction.steps_per_decision'] > 1 else None,
    #     # 'state_constructor': lambda d: ['multi_trace', ] if not d['interaction.only_dp_transitions'] else None,
    #
    #
    # }
}


# can we make conditionals depend on other conditionals?
