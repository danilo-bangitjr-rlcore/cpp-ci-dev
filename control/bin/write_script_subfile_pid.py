import copy

from write_script import combinations, merge_independent
"""
Named as test_v1.
After commit cb79808
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

    # shared_settings["--action_normalizer"] = ["Scale"]
    # shared_settings["--activation"] = ["ReLU6"]
    # shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/replay1_batch1/beta_shift_0/activation_relu6"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--activation"] = ["ReLU"]
    shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/replay1_batch1/beta_shift_0/activation_relu"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

def c20240117_0(settings, shared_settings, target_agents):
    """
    Recreate directly-learn-beta-parameter result (commit before merging on the main thread)
    Remove environment scaler, use beta distribution scaler=10
    """
    """
    A good command line
    python3 main.py
        --agent_name GAC
        --rho 0.1
        --prop_rho_mult 2
        --version 1
        --max_steps 5000
        --debug 0
        --render 0
        --polyak 0
        --env_action_scaler 1.0
        --head_activation ReLU
        --hidden_actor 0
        --optimizer RMSprop
        --env_name NonContexTT
        --env_info 1.0 3.0
        --buffer_size 1
        --batch_size 1
        --beta_parameter_bias 0.0
        --tau 0.001
        --layer_init_actor Const/10/0
        --exp_name temp/
        --exp_info /totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/
        --lr_actor 0.5
        --lr_critic 0.00001
        --action_normalizer Scale
        --activation ReLU6
        --seed 0
    """
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--env_info"] = [[1., 3.]]
    shared_settings["--debug"] = [1]
    shared_settings["--render"] = [2]
    shared_settings["--buffer_size"] = [1]
    shared_settings["--batch_size"] = [1]
    shared_settings["--beta_parameter_bias"] = [0.]
    shared_settings["--tau"] = [1e-3]
    shared_settings["--hidden_actor"] = [[0]]
    shared_settings["--layer_init_actor"] = ["Const/10/0"]
    shared_settings["--exp_name"] = ["recreating_results_vis"]
    shared_settings["--lr_actor"] = [0.5]
    shared_settings["--lr_critic"] = [1e-5]
    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--optimizer"] = ["RMSprop"]
    shared_settings["--activation"] = ["ReLU6"]

    shared_settings["--exp_info"] = ["/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    shared_settings["--activation"] = ["ReLU"]
    shared_settings["--exp_info"] = ["/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1, comb_num_base=0)

    shared_settings["--optimizer"] = ["Adam"]
    shared_settings["--exp_info"] = ["/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU/change_to_Adam"] # fails in this setting
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=2, line_per_file=1, comb_num_base=0)

    shared_settings["--optimizer"] = ["RMSprop"]
    shared_settings["--action_normalizer"] = ["Identity"]
    shared_settings["--exp_info"] = ["/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU/remove_action_normalizer"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=3, line_per_file=1, comb_num_base=0)

def c20240118(settings, shared_settings, target_agents):
    """
    Recreate nonlinear beta learning result
    Remove environment scaler, use beta distribution scaler=10
    """
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--env_action_scaler"] = [1]
    shared_settings["--env_info"] = [[1., 3.]]
    # shared_settings["--debug"] = [1]
    # shared_settings["--render"] = [2]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    shared_settings["--buffer_prefill"] = [32]
    shared_settings["--beta_parameter_bias"] = [1.]
    shared_settings["--tau"] = [0]
    shared_settings["--rho"] = [0.4]
    shared_settings["--prop_rho_mult"] = [2.0]
    shared_settings["--exp_name"] = ["recreating_results_vis"]
    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--optimizer"] = ["RMSprop"]
    shared_settings["--activation"] = ["ReLU6"]
    shared_settings["--head_activation"] = ["Softplus"]

    shared_settings["--lr_actor"] = [2.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]
    shared_settings["--lr_critic"] = [0.03, 0.001, 0.0003, 0.0001, 3e-5, 1e-5, 3e-6]

    shared_settings["--exp_info"] = ["/sweep_obs1/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

def c20240119(settings, shared_settings, target_agents):
    """
    Recreate nonlinear beta learning result
    Remove environment scaler, use beta distribution scaler=10
    """
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--env_action_scaler"] = [1]
    shared_settings["--env_info"] = [[0., 3.]]
    # shared_settings["--debug"] = [1]
    # shared_settings["--render"] = [2]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    shared_settings["--buffer_prefill"] = [32]
    shared_settings["--beta_parameter_bias"] = [1.]
    shared_settings["--tau"] = [0]
    shared_settings["--rho"] = [0.4]
    shared_settings["--prop_rho_mult"] = [2.0]
    shared_settings["--exp_name"] = ["recreating_results_vis"]
    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--optimizer"] = ["RMSprop"]
    shared_settings["--activation"] = ["ReLU6"]
    shared_settings["--head_activation"] = ["Softplus"]

    # shared_settings["--lr_actor"] = [1.0]
    # shared_settings["--lr_critic"] = [1e-5]
    # shared_settings["--exp_info"] = ["/recrete_nonlinear_test/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    shared_settings["--rho"] = [0.1]
    shared_settings["--prop_rho_mult"] = [4.0, 8.0]
    shared_settings["--lr_actor"] = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001, 0.00003, 0.00001]
    shared_settings["--lr_critic"] = [0.03, 0.001, 0.0003, 0.0001, 3e-5, 1e-5]
    shared_settings["--exp_info"] = ["/sweep_obs0/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    shared_settings["--rho"] = [0.4]
    shared_settings["--prop_rho_mult"] = [2.0]
    shared_settings["--env_info"] = [[1., 3.]]
    shared_settings["--lr_actor"] = [0.00003, 0.00001]
    shared_settings["--lr_critic"] = [0.03, 0.001, 0.0003, 0.0001, 3e-5, 1e-5, 3e-6]
    shared_settings["--exp_info"] = ["/sweep_obs1/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=108, line_per_file=1, comb_num_base=56)


# def c20240116_1(settings, shared_settings, target_agents):
#     """
#     Move the setting to contextual bandit
#     """
#     shared_settings["--env_name"] = ["ThreeTank"]
#     shared_settings["--env_info"] = [[1]+list(range(2, 5))]
#     shared_settings["--state_normalizer"] = ["OneHot"]
#     shared_settings["--buffer_size"] = [100]
#     shared_settings["--batch_size"] = [32]
#     shared_settings["--beta_parameter_bias"] = [1.]
#     shared_settings["--hidden_actor"] = [[16]]
#     shared_settings["--hidden_critic"] = [[16, 16]]
#     shared_settings["--exp_name"] = ["small_network"]
#     shared_settings["--exp_info"] = ["/setpoint_2-4/obs_onehot/replay100_batch32/beta_shift_1/"]
#
#     shared_settings["--lr_actor"] = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
#     shared_settings["--lr_critic"] = [0.001, 0.0003, 0.0001, 3e-5, 1e-5]
#     settings = merge_independent(settings, shared_settings)
#     combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

if __name__=='__main__':
    settings = {
        "GAC": {
            "--rho": [0.1],
            "--prop_rho_mult": [2],
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
    target_agents = ["GAC"]

    c20240117_0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
