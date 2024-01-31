import copy

from write_script import combinations, merge_independent
    
       
def reactor_env_sweep():
    """
    Determining best GAC hyperparameter values in ReactorEnv to try to determine sensible values for
    Reseau pilot plant
    
    Ran on Jan 26, 2024
    """
    settings = {
        "GAC": {
            "--rho": [0.4],
            "--prop_rho_mult": [1.5, 2.0],
            "--n": [30],
            "--buffer_size": [30000],
            "--buffer_prefill": [0, 256],
            "--batch_size": [256, 1024],
            "--polyak": [0.995],
            "--lr_actor": [1e-2, 1e-3, 1e-4],
            "--lr_critic": [1e-2, 1e-3, 1e-4],
            "--beta_parameter_bias": [0.0, 1.0],
        },
    }
    shared_settings = {
        "--env_name": ["ReactorEnv"],
        "--exp_name": ["Restricted_Param_Sweep"],
        "--max_steps": [30000],
        "--timeout": [100],
        "--gamma": [0.99],
        "--hidden_actor": ["256 256"],
        "--hidden_critic": ["256 256"],
        "--log_interval": [10],
        "--stats_queue_size": [1],
        "--state_normalizer": ["Identity"],
        "--action_normalizer": ["Scale"],
        "--reward_normalizer": ["Identity"],
        "--actor": ["Beta"],
        "--critic": ["FC"],
        "--optimizer": ["RMSprop"],
        "--action_scale": [2],
        "--action_bias": [-1],
        "--debug": [0],
        "--render": [0],
    }
    target_agents = ["GAC"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=2, line_per_file=10000, comb_num_base=432)
    
    
if __name__=='__main__':
    reactor_env_sweep()