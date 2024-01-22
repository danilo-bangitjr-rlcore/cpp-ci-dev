import copy

from write_script import combinations, merge_independent
"""
Named as test_v1.
After commit cb79808
"""

def c20240122(settings, shared_settings, target_agents):
    """
    Recreate directly-learn-beta-parameter result (commit before merging on the main thread)
    Remove environment scaler, use beta distribution scaler=10
    """
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--env_info"] = [[1., 3.]]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [256]
    shared_settings["--beta_parameter_bias"] = [1.]
    shared_settings["--hidden_actor"] = [[256, 256]]
    shared_settings["--layer_init_actor"] = ["Const/10/0"]
    shared_settings["--exp_name"] = ["heuristic_lr_exp"]
    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--activation"] = ["ReLU"]
    shared_settings["--lr_actor"] = [0.1]
    shared_settings["--lr_critic"] = [0.03]
    shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/replay5000_batch256/beta_shift_1/activation_relu"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

if __name__=='__main__':
    settings = {
        "LineSearch": {
            "--rho": [0.1],
            "--tau": [1e-3],
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
        "--optimizer": ["Adam"],
        "--action_normalizer": ["Scale"],
    }
    target_agents = ["LineSearch"]

    c20240122(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
