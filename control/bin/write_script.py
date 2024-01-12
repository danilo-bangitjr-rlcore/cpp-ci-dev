import os
import numpy as np
import itertools
from pathlib import Path


def base_cmd(**kwargs):
    cmd = "python3 main.py "
    for k in kwargs:
        cmd += " {} {} ".format(k, kwargs[k])
    if "/" in kwargs["--env_name"]:
        env_name = "_".join(kwargs["--env_name"].split("/"))
    else:
        env_name = kwargs["--env_name"]
    # cmd += "> {}_{}_{}.txt ".format("_".join(kwargs["--env_name"].split("/")), kwargs["--param"], kwargs["--seed"])
    cmd += "\n"
    return cmd

def write_cmd(cmds, prev_file=0, line_per_file=1):
    curr_dir = os.getcwd()
    cmd_file_path = os.path.join(curr_dir, "../out/scripts/tasks_{}.sh")

    cmd_file = Path(cmd_file_path.format(int(prev_file)))
    cmd_file.parent.mkdir(exist_ok=True, parents=True)

    count = 0
    print("First script:", cmd_file)
    file = open(cmd_file, 'w')
    for cmd in cmds:
        file.write(cmd)
        count += 1
        if count % line_per_file == 0:
            file.close()
            prev_file += 1
            cmd_file = Path(cmd_file_path.format(int(prev_file)))
            file = open(cmd_file, 'w')
    if not file.closed:
        file.close()
    print("Last script:", cmd_file_path.format(str(prev_file)), "\n")


def merge_independent(settings, shared_settings):
    for agent in settings:
        settings[agent].update(shared_settings)
    return settings
    
def combinations(settings, target_agents, num_runs=10, comb_num_base=0, prev_file=0, line_per_file=1):
    cmds = []
    kwargs = {}
    for agent in target_agents:
        sweep_params = settings[agent]
        keys, values = zip(*sweep_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            kwargs["--param"] = comb_num + comb_num_base
            kwargs["--agent_name"] = agent
            for (k, v) in param_comb.items():
                kwargs[k] = v
            if "--lr_actor" not in param_comb:
                kwargs["--lr_actor"] = param_comb["--lr_critic"] * 0.1
            for run in list(range(num_runs)):
                kwargs["--seed"] = run
                if "--load_dir" in param_comb:
                    kwargs["--load_dir"] = kwargs["--load_dir"].format(run)
                
                cmds.append(base_cmd(**kwargs))
    write_cmd(cmds, prev_file=prev_file, line_per_file=line_per_file)


def smpl_exp():
    settings = {
        "GAC": {
            "--tau": [1e0, 1e-1, 1e-2, 1e-3],
            "--rho": [0.05, 0.1, 0.25],
            "--n": [30],
            "--buffer_size": [1000, 10000],
            "--buffer_prefill": [1000],
            "--batch_size": [64, 256],
            "--polyak": [0.9, 0.995],
            "--lr_actor": [1e-3, 1e-4, 1e-5, 1e-6],
            "--lr_critic": [1e-2, 1e-3, 1e-4, 1e-5],
        },
        "SAC": {
            "--tau": [1e-1, 1e-2, 1e-3, 1e-4],
            "--buffer_size": [1000, 10000],
            "--buffer_prefill": [1000],
            "--batch_size": [64, 256],
            "--polyak": [0.9, 0.995],
            "--lr_actor": [1e-4],
            "--lr_critic": [1e-3],
        },
        "Reinforce": {
            "--buffer_size": [0],
            "--buffer_prefill": [0],
            "--batch_size": [1],
            "--lr_actor": [1e-2, 1e-3, 1e-4],
            "--lr_v": [1e-1, 1e-2, 1e-3]
        }
    }
    shared_settings = {
        "--env_name": ["ReactorEnv"],
        "--exp_name": ["With_LR_Param_Baseline"],
        "--exp_info": ["/Param_Sweep"],
        "--max_steps": [100000],
        "--timeout": [100],
        "--gamma": [0.9, 0.99],
        "--log_interval": [10],
        "--stats_queue_size": [1],
        "--state_normalizer": ["Identity"],
        "--reward_normalizer": ["Identity"],
        "--actor": ["Beta"],
        "--critic": ["FC"],
        "--optimizer": ["RMSprop"],
        "--hidden_actor": ["256 256"],
        "--hidden_critic": ["256 256"],
        "--action_scale": [2],
        "--action_bias": [-1],
        "--debug": [1],
    }
    target_agents = ["Reinforce"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, comb_num_base=0, line_per_file=3)

def gem_episodic_exp():
    settings = {
        "GAC": {
            "--tau": [1e-1, 1e-2, 1e-3, 1e-4],
            "--rho": [0.1, 0.25],
            "--n": [30],
            "--buffer_size": [1000, 10000],
            "--buffer_prefill": [1000],
            "--batch_size": [64, 256],
            "--polyak": [0.9, 0.995],
            "--lr_actor": [1e-2],
            "--lr_critic": [1e-3],
        },
        "SAC": {
            "--tau": [1e-1, 1e-2, 1e-3, 1e-4],
            "--buffer_size": [1000, 10000],
            "--buffer_prefill": [1000],
            "--batch_size": [64, 256],
            "--polyak": [0.9, 0.995],
            "--lr_actor": [1e-2],
            "--lr_critic": [1e-4],
        },
        "Reinforce": {
            "--buffer_size": [0],
            "--buffer_prefill": [0],
            "--batch_size": [1],
            "--lr_actor": [1e-2],
            "--lr_v": [1e-2],
        }
    }
    shared_settings = {
        "--env_name": ["Cont-CC-PermExDc-v0"],
        "--evaluation_criteria": ["return"],
        "--exp_name": ["With_LR_Param_Baseline"],
        "--exp_info": ["/Episodic/Param_Sweep"],
        "--max_steps": [50000],
        "--timeout": [200],
        "--gamma": [0.9, 0.99],
        "--log_interval": [10],
        "--stats_queue_size": [1],
        "--state_normalizer": ["Identity"],
        "--reward_normalizer": ["Identity"],
        "--actor": ["Beta"],
        "--critic": ["FC"],
        "--optimizer": ["RMSprop"],
        "--hidden_actor": ["256 256"],
        "--hidden_critic": ["256 256"],
        "--action_scale": [2],
        "--action_bias": [-1],
    }
    target_agents = ["GAC", "SAC"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=2000)

def gem_continuing_exp():
    settings = {
        "GAC": {
            "--tau": [1e-2],
            "--rho": [0.1],
            "--n": [30],
            "--buffer_size": [1000],
            "--buffer_prefill": [1000],
            "--batch_size": [64],
            "--polyak": [0.995],
            "--lr_actor": [1e-2, 1e-3, 1e-4, 1e-5],
            "--lr_critic": [1e-2, 1e-3, 1e-4, 1e-5],
        },
        "SAC": {
            "--tau": [1e-2],
            "--buffer_size": [1000],
            "--buffer_prefill": [1000],
            "--batch_size": [64],
            "--polyak": [0.995],
            "--lr_actor": [1e-2, 1e-3, 1e-4, 1e-5],
            "--lr_critic": [1e-2, 1e-3, 1e-4, 1e-5],
        },
        "Reinforce": {
            "--buffer_size": [0],
            "--buffer_prefill": [0],
            "--batch_size": [1],
            "--lr_actor": [1e-2, 1e-3, 1e-4, 1e-5],
            "--lr_v": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    }
    shared_settings = {
        "--env_name": ["Cont-CC-PermExDc-v0"],
        "--evaluation_criteria": ["avg_reward"],
        "--reward_window": [1000],
        "--exp_name": ["With_LR_Param_Baseline"],
        "--exp_info": ["/Continuing/Param_Sweep"],
        "--max_steps": [50000],
        "--timeout": [100000],
        "--gamma": [0.9, 0.99, 0.999],
        "--log_interval": [10],
        "--stats_queue_size": [1],
        "--state_normalizer": ["Identity"],
        "--reward_normalizer": ["Identity"],
        "--actor": ["Beta"],
        "--critic": ["FC"],
        "--optimizer": ["RMSprop"],
        "--hidden_actor": ["256 256"],
        "--hidden_critic": ["256 256"],
        "--action_scale": [2],
        "--action_bias": [-1],
    }
    target_agents = ["GAC", "SAC", "Reinforce"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=2000)

def etc_pid():
    settings = {
        "ETC": {
            "--actions_per_dim": [32],
            "--min_trials": [1],
            "--render": [0],
            "--lr_actor": [1e-3],
            "--buffer_size": [1],
            "--batch_size": [1],
        },
        "GAC": {
<<<<<<< HEAD
            "--tau": [0],
            "--rho": [0.1, 0.2, 0.3],
            "--prop_rho_mult": [1.0, 2.0, 3.0],
            "--n": [30],
            "--buffer_size": [100],
            "--batch_size": [32],
            "--polyak": [0.0],
            "--lr_actor": [1e1, 1e0, 1e-1, 1e-2],
            "--lr_critic": [1e-3, 1e-4, 1e-5, 1e-6],
=======
            "--tau": [1e0, 1e-1, 1e-2, 1e-3],
            "--rho": [0.1, 0.25],
            "--n": [30],
            "--buffer_size": [100],
            "--batch_size": [32],
            "--polyak": [0.0, 0.995],
            "--lr_actor": [1e-1, 1e-2, 1e-3, 1e-4],
            "--lr_critic": [1e-6],
>>>>>>> f097922ff9bbeb2efd18d19a48495c64c64e62ee
            "--render": [2],
        },
    }
    shared_settings = {
        "--env_name": ["NonContexTT"],
<<<<<<< HEAD
        "--exp_name": ["Noncontext_PID_Action_Visits"],
=======
        "--exp_name": ["Noncontext_PID_Visit_Heatmap"],
>>>>>>> f097922ff9bbeb2efd18d19a48495c64c64e62ee
        "--exp_info": ["/"],
        "--evaluation_criteria": ["return"],
        "--debug": [1],
        "--max_steps": [10],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
    }
    target_agents = ["GAC"]

    settings = merge_independent(settings, shared_settings)
<<<<<<< HEAD
    combinations(settings, target_agents, num_runs=1, prev_file=8, line_per_file=18, comb_num_base=144)
=======
    combinations(settings, target_agents, num_runs=1, prev_file=10, line_per_file=16, comb_num_base=416)
>>>>>>> f097922ff9bbeb2efd18d19a48495c64c64e62ee

def test_runs():
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
        "--exp_name": ["temp"],
        "--max_steps": [2],
        "--render": [0],
        "--lr_actor": [0.001],  # [0.01, 0.001, 0.0001],
        "--lr_critic": [0.001],  # [0.01, 0.001, 0.0001],
        "--buffer_size": [10],
        "--batch_size": [5],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        
        "--hidden_actor": ["200"],
        "--hidden_critic": ["100"],
        "--activation": ["ReLU"],
    }
    target_agents = ["SimpleAC", "SAC", "GAC"]
    
    shared_settings["--env_name"] = ["ThreeTank", "TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/action_0_1/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)
    
    shared_settings["--env_name"] = ["ThreeTank", "TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/action_0_4/"]
    shared_settings["--action_scale"] = [4]
    shared_settings["--action_bias"] = [0]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=6, line_per_file=1)
    
    shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    shared_settings["--exp_info"] = ["/target0/replay0/action_-0.1_0.1/"]
    shared_settings["--action_scale"] = [0.2]
    shared_settings["--action_bias"] = [-0.1]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=12, line_per_file=1)
    
    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.1]
    shared_settings["--exp_info"] = ["/target0/replay0/change_0.1"]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [3]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=15, line_per_file=1)
    
    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.01]
    shared_settings["--exp_info"] = ["/target0/replay0/change_0.01"]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [3]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1)


if __name__ == '__main__':
    # test_runs()
    # demo()
    # constant_pid() # 52919
    # smpl_exp()
    # gem_episodic_exp()
    etc_pid()
