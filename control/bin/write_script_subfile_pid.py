import copy

from write_script import combinations, merge_independent

def demo():
    settings = {
        "GAC": {
            "--tau": [1e-3],
            "--rho": [0.1],
        },
    }
    shared_settings = {
        "--exp_name": ["demo"],
        "--max_steps": [7000],
        "--render": [0],
        "--lr_actor": [0.001, 0.0001],
        "--lr_critic": [0.001, 0.0001],
        "--buffer_size": [1],
        "--batch_size": [1],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
    }
    target_agents = ["GAC"]
    shared_settings["--env_name"] = ["ThreeTank"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=5, prev_file=0, line_per_file=1)

def stable_gac_test(settings, shared_settings, target_agents):
    settings = {
        "GAC": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
        "GACMH": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
        "GACPS": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
        "GACIn": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
    }
    shared_settings = {
        "--exp_name": ["stable_gac_test/v0/"],
        "--max_steps": [5000],
        "--render": [0],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--buffer_size": [50],
        "--batch_size": [8],

        "--tau": [1e-2, 1e-3, 1e-4],
        "--rho": [0.1, 0.2],
        "--lr_actor": [0.01, 0.001, 0.0001],
        "--lr_critic": [0.01, 0.001, 0.0001],
    }

    def clip_action(settings, shared_settings):
        target_agents = ["GAC"]
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["target0/replay5000_batch8/env_scale_10_action_0.01_0.99/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--action_scale"] = [0.99]
        shared_settings["--action_bias"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

        # 2
        shared_settings["--env_name"] = ["TTAction/ConstPID"]

        shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10_action_0.01_0.99/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--action_scale"] = [0.99]
        shared_settings["--action_bias"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=54, line_per_file=1)


    def reward_stay(settings, shared_settings):
        target_agents = ["GAC"]
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteRwdStay"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)

        shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/change_0.01/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def batch_normalization(settings, shared_settings):
        target_agents = ["GAC"]
        shared_settings["--layer_norm"] = [1]

        # # 1
        # shared_settings["--env_name"] = ["NonContexTT"]
        #
        # shared_settings["--exp_info"] = ["/target0_batchNorm/replay5000_batch8/env_scale_10/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

        # 2
        shared_settings["--env_name"] = ["TTAction/ConstPID"]

        shared_settings["--exp_info"] = ["/target0_batchNorm/replay5000_batch8/env_scale_10/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=54, line_per_file=1)

        # # 3
        # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--action_scale"] = [0.2]
        # shared_settings["--action_bias"] = [-0.1]
        #
        # shared_settings["--exp_info"] = ["/target0_batchNorm/replay5000_batch8/env_scale_1/action_-0.1_0.1/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=108, line_per_file=1)
        #
        # # 4
        # shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        # shared_settings["--env_info"] = [0.01]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--actor"] = ["Softmax"]
        # shared_settings["--discrete_control"] = [1]
        # shared_settings.pop('--action_scale', None)
        # shared_settings.pop('action_bias', None)
        #
        # shared_settings["--exp_info"] = ["/target0_batchNorm/replay5000_batch8/env_scale_1/change_0.01/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=162, line_per_file=1)


    def gac_with_memory(settings, shared_settings):
        target_agents = ["GACMH"]

        # # 1
        # shared_settings["--env_name"] = ["NonContexTT"]
        #
        # shared_settings["--exp_info"] = ["/target0/replay50_batch8/env_scale_10/"]
        # shared_settings["--buffer_size"] = [50]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

        # 2
        shared_settings["--env_name"] = ["TTAction/ConstPID"]

        shared_settings["--exp_info"] = ["/target0/replay50_batch8/env_scale_10/"]
        shared_settings["--buffer_size"] = [50]
        shared_settings["--batch_size"] = [8]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=6)

        # It won't work for changed action here.
        # So we only do test in Noncontextual and Direct action setting

    def gac_predict_success(settings, shared_settings):
        target_agents = ["GACPS"]
        shared_settings["--rho"] = [0.1, 0.2, 0.5, 1]
        shared_settings["--exp_name"] = ["stable_gac_test/v1/"]

        # 1
        shared_settings["--env_name"] = ["NonContexTT"]

        shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

        # 2
        shared_settings["--env_name"] = ["TTAction/ConstPID"]

        shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=108, line_per_file=1)

        # # 3
        # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--action_scale"] = [0.2]
        # shared_settings["--action_bias"] = [-0.1]
        #
        # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=108, line_per_file=1)
        #
        # # 4
        # shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        # shared_settings["--env_info"] = [0.01]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--actor"] = ["Softmax"]
        # shared_settings["--discrete_control"] = [1]
        # shared_settings.pop('--action_scale', None)
        # shared_settings.pop('action_bias', None)
        #
        # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/change_0.01/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=162, line_per_file=1)

    def gac_inac(settings, shared_settings):
        target_agents = ["GACIn"]

        # # 1
        # shared_settings["--env_name"] = ["NonContexTT"]
        #
        # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=216, line_per_file=1)

        # 2
        shared_settings["--env_name"] = ["TTAction/ConstPID"]

        shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=6)

        # # 3
        # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--action_scale"] = [0.2]
        # shared_settings["--action_bias"] = [-0.1]
        #
        # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=324, line_per_file=1)
        #
        # # 4
        # shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        # shared_settings["--env_info"] = [0.01]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--actor"] = ["Softmax"]
        # shared_settings["--discrete_control"] = [1]
        # shared_settings.pop('--action_scale', None)
        # shared_settings.pop('action_bias', None)
        #
        # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/change_0.01/"]
        # shared_settings["--buffer_size"] = [5000]
        # shared_settings["--batch_size"] = [8]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=378, line_per_file=1)

    clip_action(copy.deepcopy(settings), copy.deepcopy(shared_settings))
    # reward_stay(copy.deepcopy(settings), copy.deepcopy(shared_settings))
    # batch_normalization(copy.deepcopy(settings), copy.deepcopy(shared_settings))
    # gac_with_memory(copy.deepcopy(settings), copy.deepcopy(shared_settings))
    # gac_predict_success(copy.deepcopy(settings), copy.deepcopy(shared_settings))
    # gac_inac(copy.deepcopy(settings), copy.deepcopy(shared_settings))

def gac_sweep(settings, shared_settings, target_agents):
    settings = {
        "GAC": {
        },
    }
    shared_settings = {
        "--exp_name": ["parameter_study"],
        "--max_steps": [5000],
        "--render": [0],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],

        "--tau": [1e-2, 1e-3, 1e-4],
        "--rho": [0.1, 0.2],
        "--lr_actor": [0.01, 0.001, 0.0001],
        "--lr_critic": [0.01, 0.001, 0.0001],
    }
    target_agents = ["GAC"]

    # # 1
    shared_settings["--env_name"] = ["NonContexTT"]
    #
    # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_10/"]
    # shared_settings["--buffer_size"] = [100]
    # shared_settings["--batch_size"] = [32]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay50_batch16/env_scale_10/"]
    # shared_settings["--buffer_size"] = [50]
    # shared_settings["--batch_size"] = [16]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=54, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay50_batch8/env_scale_10/"]
    # shared_settings["--buffer_size"] = [50]
    # shared_settings["--batch_size"] = [8]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=108, line_per_file=1)

    # shared_settings["--exp_info"] = ["/target0/replay5000_batch32/env_scale_10/"]
    # shared_settings["--buffer_size"] = [5000]
    # shared_settings["--batch_size"] = [32]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    # shared_settings["--buffer_size"] = [5000]
    # shared_settings["--batch_size"] = [8]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=54, line_per_file=1)

    # 4
    shared_settings["--env_name"] = ["TTAction/ConstPID"]

    shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_10/"]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=27, line_per_file=6)

    shared_settings["--exp_info"] = ["/target0/replay50_batch16/env_scale_10/"]
    shared_settings["--buffer_size"] = [50]
    shared_settings["--batch_size"] = [16]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=36, line_per_file=6)

    shared_settings["--exp_info"] = ["/target0/replay50_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [50]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=45, line_per_file=6)

    shared_settings["--exp_info"] = ["/target0/replay5000_batch32/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [32]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=54, line_per_file=6)

    shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=63, line_per_file=6)

    # # 2
    # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--action_scale"] = [0.2]
    # shared_settings["--action_bias"] = [-0.1]
    #
    # # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"]
    # # shared_settings["--buffer_size"] = [100]
    # # shared_settings["--batch_size"] = [32]
    # # settings = merge_independent(settings, shared_settings)
    # # combinations(settings, target_agents, num_runs=1, prev_file=162, line_per_file=1)
    # #
    # # shared_settings["--exp_info"] = ["/target0/replay50_batch16/env_scale_1/action_-0.1_0.1/"]
    # # shared_settings["--buffer_size"] = [50]
    # # shared_settings["--batch_size"] = [16]
    # # settings = merge_independent(settings, shared_settings)
    # # combinations(settings, target_agents, num_runs=1, prev_file=216, line_per_file=1)
    # #
    # # shared_settings["--exp_info"] = ["/target0/replay50_batch8/env_scale_1/action_-0.1_0.1/"]
    # # shared_settings["--buffer_size"] = [50]
    # # shared_settings["--batch_size"] = [8]
    # # settings = merge_independent(settings, shared_settings)
    # # combinations(settings, target_agents, num_runs=1, prev_file=270, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay5000_batch32/env_scale_1/action_-0.1_0.1/"]
    # shared_settings["--buffer_size"] = [5000]
    # shared_settings["--batch_size"] = [32]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=216, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"]
    # shared_settings["--buffer_size"] = [5000]
    # shared_settings["--batch_size"] = [8]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=270, line_per_file=1)
    #
    # # 3
    # shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    # shared_settings["--env_info"] = [0.01]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--actor"] = ["Softmax"]
    # shared_settings["--discrete_control"] = [1]
    # shared_settings.pop('--action_scale', None)
    # shared_settings.pop('action_bias', None)
    # #
    # # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_1/change_0.01/"]
    # # shared_settings["--buffer_size"] = [100]
    # # shared_settings["--batch_size"] = [32]
    # # settings = merge_independent(settings, shared_settings)
    # # combinations(settings, target_agents, num_runs=1, prev_file=324, line_per_file=1)
    # #
    # # shared_settings["--exp_info"] = ["/target0/replay50_batch16/env_scale_1/change_0.01/"]
    # # shared_settings["--buffer_size"] = [50]
    # # shared_settings["--batch_size"] = [16]
    # # settings = merge_independent(settings, shared_settings)
    # # combinations(settings, target_agents, num_runs=1, prev_file=378, line_per_file=1)
    # #
    # # shared_settings["--exp_info"] = ["/target0/replay50_batch8/env_scale_1/change_0.01/"]
    # # shared_settings["--buffer_size"] = [50]
    # # shared_settings["--batch_size"] = [8]
    # # settings = merge_independent(settings, shared_settings)
    # # combinations(settings, target_agents, num_runs=1, prev_file=432, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay5000_batch32/env_scale_1/change_0.01/"]
    # shared_settings["--buffer_size"] = [5000]
    # shared_settings["--batch_size"] = [32]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=324, line_per_file=1)
    #
    # shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_1/change_0.01/"]
    # shared_settings["--buffer_size"] = [5000]
    # shared_settings["--batch_size"] = [8]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=378, line_per_file=1)

def gac_proposal_wo_entropy(settings, shared_settings, target_agents):
    settings = {
        "GAC": {
        },
    }
    shared_settings = {
        "--exp_name": ["GAC_proposal_wo_entropy"],
        "--max_steps": [5000],
        "--render": [0],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],

        "--tau": [0],
        "--rho": [0.1],
        "--lr_actor": [0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
        "--lr_critic": [0.01, 0.003, 0.001, 0.0003, 0.0001, 3e-5, 1e-5],
    }
    target_agents = ["GAC"]

    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=49, line_per_file=1, comb_num_base=0)

def larger_range_learning_rate_sweep(settings, shared_settings, target_agents):
    settings = {
        "GAC": {
        },
    }
    shared_settings = {
        "--exp_name": ["learning_rate_larger_range"],
        "--max_steps": [5000],
        "--render": [0],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],

        "--tau": [1e-3],
        "--rho": [0.1],
        "--lr_actor": [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.1, 0.3, 0.5, 1.0],
        "--lr_critic": [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 3e-5, 1e-5],
    }
    target_agents = ["GAC"]

    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=64)

    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay5000_batch8/env_scale_10/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1, comb_num_base=64)

def constant_pid_target0_replay0(settings, shared_settings, target_agents):
    """
    Without replay
    """
    # # debugging step
    # shared_settings["--env_name"] = ["ThreeTank"]
    # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    # shared_settings["--env_name"] = ["NonContexTT"]
    # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=72, line_per_file=6)

    # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/action_-0.1_0.1/"]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--action_scale"] = [0.2]
    # shared_settings["--action_bias"] = [-0.1]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=9, line_per_file=1)
    #
    # shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    # shared_settings["--env_info"] = [0.01]
    # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/change_0.01"]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--actor"] = ["Softmax"]
    # shared_settings["--discrete_control"] = [1]
    # shared_settings.pop('--action_scale', None)
    # shared_settings.pop('action_bias', None)
    # settings["GAC"]["--n"] = [30]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1)

def constant_pid_target0_replay0_clip_distribution_param(settings, shared_settings, target_agents):
    shared_settings["--exp_name"] = ["clip_distribution_param"]
    shared_settings["--head_activation"] = ["ReLU6"]

    """
    Without replay
    """
    # shared_settings["--env_name"] = ["ThreeTank"]
    # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=3)

    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=82, line_per_file=6)

    # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/action_-0.1_0.1/"]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--action_scale"] = [0.2]
    # shared_settings["--action_bias"] = [-0.1]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1)

def constant_pid_target0_replay0_clip_action(settings, shared_settings, target_agents):
    """
    Without replay
    """
    shared_settings["--env_name"] = ["TTChangeAction/ClipConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/action_-0.1_0.1/"]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--action_scale"] = [0.2]
    shared_settings["--action_bias"] = [-0.1]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    shared_settings["--env_name"] = ["TTChangeAction/ClipDiscreteConstPID"]
    shared_settings["--env_info"] = [0.01]
    shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/change_0.01"]
    shared_settings["--env_action_scaler"] = [1.]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [30]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=27, line_per_file=1)

def constant_pid_target0_replay100(settings, shared_settings, target_agents):
    """
    With replay size 100, batch size 32
    """

    # # Debugging step
    # shared_settings["--env_name"] = ["ThreeTank"]
    # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_10/"]
    # shared_settings["--buffer_size"] = [100]
    # shared_settings["--batch_size"] = [32]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=4, line_per_file=1)

    # shared_settings["--env_name"] = ["NonContexTT"]
    # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_10/"]
    # shared_settings["--env_action_scaler"] = [10.]
    # shared_settings["--buffer_size"] = [100]
    # shared_settings["--batch_size"] = [32]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=9, line_per_file=1)

    shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_10/"]
    shared_settings["--env_action_scaler"] = [10.]
    shared_settings["--buffer_size"] = [100]
    shared_settings["--batch_size"] = [32]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=77, line_per_file=6)

    # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"]
    # shared_settings["--buffer_size"] = [100]
    # shared_settings["--batch_size"] = [32]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--action_scale"] = [0.2]
    # shared_settings["--action_bias"] = [-0.1]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=36, line_per_file=1)
    #
    # shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    # shared_settings["--env_info"] = [0.01]
    # shared_settings["--exp_info"] = ["/target0/replay100_batch32/env_scale_1/change_0.01"]
    # shared_settings["--buffer_size"] = [100]
    # shared_settings["--batch_size"] = [32]
    # shared_settings["--env_action_scaler"] = [1.]
    # shared_settings["--actor"] = ["Softmax"]
    # shared_settings["--discrete_control"] = [1]
    # shared_settings.pop('--action_scale', None)
    # shared_settings.pop('action_bias', None)
    # settings["GAC"]["--n"] = [30]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=45, line_per_file=1)

def visualize_general(settings, shared_settings):

    def change_action_continuous_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--action_scale"] = [0.2]
        shared_settings["--action_bias"] = [-0.1]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def change_action_discrete_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        settings["GAC"]["--n"] = [30]
        shared_settings["--lr_actor"] = [0.001]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def direct_action_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay0/"]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def direct_action_replay100(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay100/"]
        shared_settings["--buffer_size"] = [100]
        shared_settings["--batch_size"] = [32]
        shared_settings["--lr_actor"] = [0.001]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=2)

    def clip_distribution_replay0(settings, shared_settings, target_agents):
        # shared_settings["--env_name"] = ["ThreeTank"]
        # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=3)

        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/target0/replay0/env_scale_10/"]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=3, line_per_file=1)

        # shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        # shared_settings["--exp_info"] = ["/target0/replay0/env_scale_1/action_-0.1_0.1/"]
        # shared_settings["--env_action_scaler"] = [1.]
        # shared_settings["--action_scale"] = [0.2]
        # shared_settings["--action_bias"] = [-0.1]
        # settings = merge_independent(settings, shared_settings)
        # combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1)

    shared_settings["--debug"] = [1]
    shared_settings["--exp_name"] = ["/visualize/"]
    shared_settings["--exp_info"] = ["/"]
    target_agents = ["GAC"]

    # change_action_continuous_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_discrete_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    direct_action_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    direct_action_replay100(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # clip_distribution_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))

def visualize_gac(settings, shared_settings):
    def noncontext_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["/replay0/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [1]
        shared_settings["--batch_size"] = [1]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.0001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def noncontext_replay50(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["/replay50/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [50]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.001]
        shared_settings["--rho"] = [0.1]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.0001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1)

    def noncontext_replay100(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["/replay100/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [100]
        shared_settings["--batch_size"] = [32]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1)

    def noncontext_replay5000(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.1]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.0001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1)

    def direct_action_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay0/"]
        shared_settings["--buffer_size"] = [1]
        shared_settings["--batch_size"] = [1]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def direct_action_replay50(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay50/"]
        shared_settings["--buffer_size"] = [50]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1)

    def direct_action_replay100(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay100/"]
        shared_settings["--buffer_size"] = [100]
        shared_settings["--batch_size"] = [32]
        shared_settings["--lr_actor"] = [0.001]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=2, line_per_file=1)

    def direct_action_replay5000(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.001]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.0001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=3, line_per_file=1)

    def direct_action_replay50_batch16(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay50_batch16/"]
        shared_settings["--buffer_size"] = [50]
        shared_settings["--batch_size"] = [16]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.001]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)


    def change_action_continuous_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay0/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--action_scale"] = [0.2]
        shared_settings["--action_bias"] = [-0.1]
        shared_settings["--buffer_size"] = [1]
        shared_settings["--batch_size"] = [1]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=4, line_per_file=1)

    def change_action_continuous_replay50(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay50/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--action_scale"] = [0.2]
        shared_settings["--action_bias"] = [-0.1]
        shared_settings["--buffer_size"] = [50]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.0001]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=3, line_per_file=1)

    def change_action_continuous_replay100(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay100/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--action_scale"] = [0.2]
        shared_settings["--action_bias"] = [-0.1]
        shared_settings["--buffer_size"] = [100]
        shared_settings["--batch_size"] = [32]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=5, line_per_file=1)

    def change_action_continuous_replay5000(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--action_scale"] = [0.2]
        shared_settings["--action_bias"] = [-0.1]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=3, line_per_file=1)

    def change_action_discrete_replay0(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--exp_info"] = ["/replay0/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        shared_settings["--lr_actor"] = [0.001]
        shared_settings["--lr_critic"] = [0.01]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=6, line_per_file=1)

    def change_action_discrete_replay50(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--exp_info"] = ["/replay50/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        shared_settings["--buffer_size"] = [50]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.0001]
        shared_settings["--rho"] = [0.1]
        shared_settings["--lr_actor"] = [0.0001]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=4, line_per_file=1)

    def change_action_discrete_replay100(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--exp_info"] = ["/replay100/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        shared_settings["--buffer_size"] = [100]
        shared_settings["--batch_size"] = [32]
        shared_settings["--lr_actor"] = [0.0001]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=7, line_per_file=1)

    def change_action_discrete_replay5000(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.0001]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=4, line_per_file=1)

    def change_action_rwd_stay_replay5000(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteRwdStay"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.0001]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def change_action_rwd_stay_long(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["TTChangeAction/DiscreteRwdStay"]
        shared_settings["--env_info"] = [0.01]
        shared_settings["--max_steps"] = [50000]
        shared_settings["--exp_info"] = ["/replay50000/"]
        shared_settings["--env_action_scaler"] = [1.]
        shared_settings["--actor"] = ["Softmax"]
        shared_settings["--discrete_control"] = [1]
        shared_settings.pop('--action_scale', None)
        shared_settings.pop('action_bias', None)
        shared_settings["--buffer_size"] = [50000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.0001]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

    def predict_goodness(settings, shared_settings, target_agents):
        target_agents = ["GACPS"]
        # 1
        shared_settings["--env_name"] = ["NonContexTT"]

        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.0001]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

        # 2
        shared_settings["--env_name"] = ["TTAction/ConstPID"]

        shared_settings["--exp_info"] = ["/replay5000/"]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--tau"] = [0.01]
        shared_settings["--rho"] = [0.2]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [0.001]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1)

    def gac_large_lr(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["/replay5000_large_lr/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--lr_actor"] = [1.]
        shared_settings["--lr_critic"] = [1e-5]
        shared_settings["--tau"] = [1e-3]
        shared_settings["--rho"] = [1e-1]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)

        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay5000_large_lr/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--lr_actor"] = [0.01]
        shared_settings["--lr_critic"] = [3e-4]
        shared_settings["--tau"] = [1e-3]
        shared_settings["--rho"] = [1e-1]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=1, line_per_file=1)

    def gac_proposal_wo_entropy(settings, shared_settings, target_agents):
        shared_settings["--env_name"] = ["NonContexTT"]
        shared_settings["--exp_info"] = ["/replay5000_wo_entropy/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--lr_actor"] = [0.5]
        shared_settings["--lr_critic"] = [3e-5]
        shared_settings["--tau"] = [0]
        shared_settings["--rho"] = [0.1]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=2, line_per_file=1)

        shared_settings["--env_name"] = ["TTAction/ConstPID"]
        shared_settings["--exp_info"] = ["/replay5000_wo_entropy/"]
        shared_settings["--env_action_scaler"] = [10.]
        shared_settings["--buffer_size"] = [5000]
        shared_settings["--batch_size"] = [8]
        shared_settings["--lr_actor"] = [0.003]
        shared_settings["--lr_critic"] = [0.003]
        shared_settings["--tau"] = [0]
        shared_settings["--rho"] = [0.1]
        settings = merge_independent(settings, shared_settings)
        combinations(settings, target_agents, num_runs=1, prev_file=3, line_per_file=1)


    shared_settings["--debug"] = [1]
    shared_settings["--render"] = [2]
    shared_settings["--exp_name"] = ["/visualize/"]
    shared_settings["--exp_info"] = ["/"]
    target_agents = ["GAC"]




    # noncontext_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # noncontext_replay50(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # noncontext_replay100(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # noncontext_replay5000(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # direct_action_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # direct_action_replay50(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # direct_action_replay100(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # direct_action_replay5000(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # direct_action_replay50_batch16(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_continuous_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_continuous_replay50(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_continuous_replay100(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_continuous_replay5000(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_discrete_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_discrete_replay50(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_discrete_replay100(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_discrete_replay5000(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_rwd_stay_replay5000(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # change_action_rwd_stay_long(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # predict_goodness(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # predict_goodness(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    gac_large_lr(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    gac_proposal_wo_entropy(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))

def temp():
    settings = {
        "GACPS": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        },
    }
    shared_settings = {
        "--exp_name": ["/temp/"],
        "--max_steps": [5000],
        "--render": [0],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--buffer_size": [50],
        "--batch_size": [8],

        "--tau": [1e-3],
        "--rho": [0.1],
        "--lr_actor": [0.001],
        "--lr_critic": [0.001],
    }
    target_agents = ["GACPS"]

    # shared_settings["--env_name"] = ["TTAction/ConstPID"]
    shared_settings["--env_name"] = ["NonContexTT"]

    shared_settings["--exp_info"] = ["/"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=1000, line_per_file=1)


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
    # target_agents = ["GAC"]

    # demo()
    # stable_gac_test(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # gac_sweep(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # gac_proposal_wo_entropy(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # larger_range_learning_rate_sweep(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # constant_pid_target0_replay0(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # constant_pid_target0_replay100(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # # constant_pid_target0_replay0_clip_action(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # constant_pid_target0_replay0_clip_distribution_param(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))

    # visualize_general(copy.deepcopy(settings), copy.deepcopy(shared_settings))
    visualize_gac(copy.deepcopy(settings), copy.deepcopy(shared_settings))

    # temp()