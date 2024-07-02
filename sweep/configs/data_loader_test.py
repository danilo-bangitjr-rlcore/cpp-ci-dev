# Slashes for defaults list fields!
SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac'],
        'env': ['reseau'],
        'data_loader' : ['direct_action'],
        'interaction' : ['offline_anytime'],
        'state_constructor' : ['reseau_anytime'],
        'normalizer' : ['normalizer'],
        'normalizer/action_normalizer': ['scale'],
        'normalizer/reward_normalizer': ['scale'],
        'experiment.exp_name': ["Merge_Party_Alerts_Test"],
        'experiment.max_steps': [50000],
        'experiment.offline_steps': [20000],
        'experiment.load_env_obs_space_from_data': [True],
        'experiment.gamma': [0.95],
        'experiment.train_split': [0.995],
        'env.reward.penalty_weight': [10.0],
        'interaction.n_step': [0],
        'alerts.action_value.gamma': [0.8],
        'alerts.action_value.ret_perc': [0.1],
        'alerts.action_value.trace_thresh': [0.1],
        'alerts.action_value.trace_decay': [0.995],
        'alerts.gvf.gamma': [0.7, 0.8, 0.9, 0.95],
        'alerts.gvf.ret_perc': [0.1],
        'alerts.gvf.trace_thresh': [0.03, 0.05, 0.075, 0.1],
        'alerts.gvf.trace_decay': [0.99, 0.995, 0.999]
    },

    'conditional': {
        # syntax: if agent == iql, then 'agent.expectile' = [0.1, 0.5, 0.8]
        # d stands for SWEEP_PARAMS['independent']
        'interaction.obs_length': lambda d: [60] if d['env'] == 'reseau' else None,
        'interaction.steps_per_decision': lambda d: [30] if d['env'] == 'reseau' else None,
        'normalizer.action_normalizer.action_high': lambda d: [300.0] if d['normalizer/action_normalizer'] == 'scale' else None,
        'normalizer.action_normalizer.action_low': lambda d: [0.0] if d['normalizer/action_normalizer'] == 'scale' else None,
        'normalizer.reward_normalizer.reward_high': lambda d: [4.0] if d['normalizer/reward_normalizer'] == 'scale' else None,
        'normalizer.reward_normalizer.reward_low': lambda d: [-9.0] if d['normalizer/reward_normalizer'] == 'scale' else None,
        'normalizer.reward_normalizer.reward_bias': lambda d: [4.0] if d['normalizer/reward_normalizer'] == 'scale' else None,
        'agent.tau': lambda d: [0.0] if d['agent'] == 'greedy_ac' else None,
        'agent.rho': lambda d: [0.04] if d['agent'] == 'greedy_ac' else None,
        'agent.num_samples': lambda d: [500] if d['agent'] == 'greedy_ac' else None,
        'agent.uniform_proposal': lambda d: [True] if d['agent'] == 'greedy_ac' else None,
        'agent.critic.polyak': lambda d: [0.995] if d['agent'] == 'greedy_ac' else None,
        'agent.critic.critic_optimizer.lr': lambda d: [0.0003] if d['agent'] == 'greedy_ac' else None,
        'agent.critic.critic_optimizer.weight_decay': lambda d: [0.01] if d['agent'] == 'greedy_ac' else None,
        'agent.actor.actor_optimizer.lr': lambda d: [0.001] if d['agent'] == 'greedy_ac' else None,
    }
}
