import copy

from write_script import combinations, merge_independent
"""
Named as test_v1: After commit cb79808
"""

def c20240116_0(settings, shared_settings, target_agents):
    """
    Recreate directly-learn-beta-parameter result (commit before merging on the main thread)
    Remove environment scaler, use beta distribution scaler=10
    """
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--env_info"] = [[1., 3.]]
    shared_settings["--buffer_size"] = [1]
    shared_settings["--batch_size"] = [1]
    shared_settings["--beta_parameter_bias"] = [0.]
    shared_settings["--tau"] = [1e-3]
    shared_settings["--hidden_actor"] = [[0]]
    shared_settings["--layer_init_actor"] = ["Const/10/0"]
    shared_settings["--exp_name"] = ["small_network"]
    shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_raw/replay1_batch1/beta_shift_0/"]

    shared_settings["--lr_actor"] = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]
    shared_settings["--lr_critic"] = [0.03, 0.001, 0.0003, 0.0001, 3e-5, 1e-5]

    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/replay1_batch1/beta_shift_0/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

def c20240116_1(settings, shared_settings, target_agents):
    """
    Move the setting to contextual bandit
    """
    shared_settings["--env_name"] = ["ThreeTank"]
    shared_settings["--env_info"] = [[1]+list(range(2, 5))]
    shared_settings["--state_normalizer"] = ["OneHot"]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    shared_settings["--beta_parameter_bias"] = [1.]
    shared_settings["--hidden_actor"] = [[16]]
    shared_settings["--hidden_critic"] = [[16, 16]]
    shared_settings["--exp_name"] = ["small_network"]
    shared_settings["--exp_info"] = ["/setpoint_2-4/obs_onehot/replay100_batch32/beta_shift_1/"]

    shared_settings["--lr_actor"] = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
    shared_settings["--lr_critic"] = [0.001, 0.0003, 0.0001, 3e-5, 1e-5]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

if __name__=='__main__':
    settings = {
        "GAC": {
            "--rho": [0.1],
            "--prop_rho_mult": [2, 8],
        },
    }
    shared_settings = {
        "--version": [1],
        "--max_steps": [5000],
        "--debug": [0],
        "--render": [0],
        "--polyak": [0],
        "--env_action_scaler": [1.],

        "--head_activation": ["ReLU"],
        "--hidden_actor": [0],
        "--optimizer": ["Adam"],
    }
    target_agents = ["GAC"]

    c20240116_0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
