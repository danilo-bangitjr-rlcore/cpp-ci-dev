import copy

from write_script import combinations, merge_independent

def learning_rate_sweep_adam_RMSprop(settings, shared_settings, target_agents):
    """
    Sweeps over learning rates for adam and RMSprop
    """
    settings = {
        "GAC": {
        },
    }
    shared_settings = {
        "--exp_name": ["learning_rate_sweep_adam_RMS_prop"],
        "--max_steps": [5000],
        "--render": [0],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--optimizer" : ['RMSprop', 'Adam'],
        "--tau": [1e-3],
        "--rho": [0.1],
        "--lr_actor": [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001],
        "--lr_critic": [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.00001, 0.00001]
    }
    target_agents = ["GAC"]

    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1000, comb_num_base=36)
    
    
    
def buffer_prefill(settings, shared_settings, target_agents):
    """
    Sweeps over learning rates for adam and RMSprop
    """
    settings = {
        "GAC": {
        },
    }
    shared_settings = {
        "--exp_name": ["buffer_prefill"],
        "--max_steps": [5000],
        "--render": [2],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--optimizer" : ['RMSprop'],
        "--tau": [2, 1, 1e-1],
        "--rho": [0.2, 0.1],
        "--lr_actor": [1, 0.5, 0.25, 0.1,],
        "--lr_critic": [0.01, 0.001, 0.0001, 0.00001]
    }
    
    
    target_agents = ["GAC"]

    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["/target0/replay5000/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8, 32]
    shared_settings["--buffer_prefill"] = [100, 1000]
    settings = merge_independent(settings, shared_settings)
    
    
    
    combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=10000, comb_num_base=385)






if __name__=='__main__':
    settings = {
        "SAC": {
            "--tau": [-1],
        },
        "GAC": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
        "GAC-OE": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
        "GACPS": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
        "SimpleAC": {
            "--tau": [1e-3],
        },
    }
    shared_settings = {
        "--exp_name": ["learning_rate"],
        "--max_steps": [5000],
        "--debug": [0],
        "--render": [0],
        "--buffer_size": [1],
        "--batch_size": [1],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--polyak": [0],
    }
    target_agents = ["SimpleAC", "SAC", "GAC"]

    # learning_rate_sweep_adam_RMSprop(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    buffer_prefill(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
