from write_script import combinations, merge_independent


def constant_pid_target0_replay0(settings, shared_settings, target_agents):
    """
    Without replay
    """
    shared_settings["--env_name"] = ["ThreeTank"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=3)

    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=9, line_per_file=3)

    shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/action_-0.1_0.1/"]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--action_scale"] = [0.2]
    shared_settings["--action_bias"] = [-0.1]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1)

    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.01]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/change_0.01"]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [30]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=45, line_per_file=1)

def constant_pid_target0_replay100(settings, shared_settings, target_agents):
    """
    With replay size 100, batch size 32
    """
    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_10/"]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=9, line_per_file=1)

    shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--action_scale"] = [0.2]
    shared_settings["--action_bias"] = [-0.1]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.01]
    shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_1/change_0.01"]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [30]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=27, line_per_file=1)

def debug(settings, shared_settings, target_agents):
    shared_settings["--debug"] = [1]
    shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    shared_settings["--exp_name"] = ["/temp/"]
    shared_settings["--exp_info"] = ["/"]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--action_scale"] = [0.2]
    shared_settings["--action_bias"] = [-0.1]
    shared_settings["--lr_actor"] = [0.01]
    shared_settings["--lr_critic"] = [0.001]
    shared_settings["--max_steps"] = [5000]
    target_agents = ["GAC"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

if __name__=='__main__':
    settings = {
        "SAC": {
            "--tau": [-1],
        },
        "GAC": {
            "--tau": [1e-3],
            "--rho": [0.1],
        },
        "SimpleAC": {
            "--tau": [1e-3],
        },
    }
    shared_settings = {
        "--exp_name": ["learning_rate"],
        "--max_steps": [5000],
        "--render": [0],
        "--lr_actor": [0.01, 0.001, 0.0001],
        "--lr_critic": [0.01, 0.001, 0.0001],
        "--buffer_size": [1],
        "--batch_size": [1],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--polyak": [0],
    }
    target_agents = ["SimpleAC", "SAC", "GAC"]

    debug(settings, shared_settings, target_agents)
