import os
import numpy as np
import itertools
from pathlib import Path


def base_cmd(**kwargs):
    cmd = "python3 main.py "
    for k in kwargs:
        cmd += " {} {} ".format(k, kwargs[k])
    cmd += "\n"
    return cmd


def write_cmd(cmds, prev_file=0, line_per_file=1):
    curr_dir = os.getcwd()
    cmd_file_path = os.path.join(curr_dir, "../out/scripts/tasks_{}.sh")

    cmd_file = Path(cmd_file_path.format(int(prev_file)))
    cmd_file.parent.mkdir(exist_ok=True, parents=True)

    count = 0
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
    print("last script:", cmd_file_path.format(str(prev_file)))


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


def reproduce():
    settings = {
        "SimpleAC": {
            "--tau": [1e-3],
        },
    }
    shared_settings = {
        "--exp_name": ["reproduce"],
        "--env_name": ["ThreeTank"],
        "--exp_info": ["/worker_1/"],
        "--env_action_scaler": [5],
        "--action_scale": [1],
        "--action_bias": [1],
        "--max_steps": [8000],
        "--render": [0],
        "--lr_actor": [0.0001],
        "--lr_critic": [0.001],
        "--lr_constrain": [0.03],
        "--buffer_size": [10],
        "--batch_size": [10],
        "--actor": ["SGaussian"],
        "--hidden_actor": ["200"],
        "--hidden_critic": ["100"],
        "--activation": ["ReLU6"],
        "--layer_init": ["Normal"],
        "--update_freq": [10],
    }
    target_agents = ["SimpleAC"]
    
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=10, prev_file=0, line_per_file=1)


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
    shared_settings["--exp_info"] = ["/without_replay/env_scale_10/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=5, prev_file=0, line_per_file=1)


def constant_pid():
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
    }
    target_agents = ["SimpleAC", "SAC", "GAC"]
    
    # shared_settings["--env_name"] = ["ThreeTank"]
    # shared_settings["--exp_info"] = ["/without_replay/env_scale_10/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=3)
    
    # shared_settings["--env_name"] = ["TTAction/ConstPID"]
    # shared_settings["--exp_info"] = ["/without_replay/env_scale_10/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=9, line_per_file=3)
    
    shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    shared_settings["--exp_info"] = ["/without_replay/env_scale_10/action_-1_1/"]
    shared_settings["--action_scale"] = [2.]
    shared_settings["--action_bias"] = [-1.]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=3)

    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.1]
    shared_settings["--exp_info"] = ["/without_replay/change_0.1"]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [9]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=27, line_per_file=3)

    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.01]
    shared_settings["--exp_info"] = ["/without_replay/change_0.01"]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [9]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=36, line_per_file=3)

def smpl_exp():
    settings = {
        "GAC": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        }
    }
    shared_settings = {
        "--env_name": ["BeerEnv"],
        "--exp_name": ["beer_env"],
        "--exp_info": ["/test"],
        "--max_steps": [1000],
        "--timeout": [100],
        "--gamma": [0.99],
        "--log_interval": [1],
        "--stats_queue_size": [1],
        "--state_normalizer": ["Identity"],
        "--reward_normalizer": ["Identity"],
        "--actor": ["Beta"],
        "--critic": ["FC"],
        "--optimizer": ["RMSprop"],
        "--polyak": [0.1], # Unsure about this
        "--hidden_actor": ["64 64"],
        "--hidden_critic": ["64 64"],
        "--lr_actor": [0.0001], #[0.01, 0.001, 0.0001],
        "--lr_critic": [0.001],#[0.01, 0.001, 0.0001],
        "--buffer_size": [1000],
        "--batch_size": [32],
        "--action_scale": [1],
        "--action_bias": [0],
    }
    target_agents = ["GAC"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=2)

def gem_exp():
    settings = {
        "GAC": {
            "--tau": [1e-3],
            "--rho": [0.1],
            "--n": [30],
        }
    }
    shared_settings = {
        "--env_name": ["Cont-CC-PermExDc-v0"],
        "--exp_name": ["gem_test"],
        "--exp_info": ["/test"],
        "--max_steps": [1000],
        "--timeout": [100],
        "--gamma": [0.99],
        "--log_interval": [1],
        "--stats_queue_size": [1],
        "--state_normalizer": ["Identity"],
        "--reward_normalizer": ["Identity"],
        "--actor": ["Beta"],
        "--critic": ["FC"],
        "--optimizer": ["RMSprop"],
        "--polyak": [0.1], # Unsure about this
        "--hidden_actor": ["64 64"],
        "--hidden_critic": ["64 64"],
        "--lr_actor": [0.0001], #[0.01, 0.001, 0.0001],
        "--lr_critic": [0.001],#[0.01, 0.001, 0.0001],
        "--buffer_size": [1000],
        "--batch_size": [32],
        "--action_scale": [1],
        "--action_bias": [0],
    }
    target_agents = ["GAC"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=2)

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
    shared_settings["--exp_info"] = ["/without_replay/action_0_1/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1)
    
    shared_settings["--env_name"] = ["ThreeTank", "TTAction/ConstPID"]
    shared_settings["--exp_info"] = ["/without_replay/action_0_4/"]
    shared_settings["--action_scale"] = [4]
    shared_settings["--action_bias"] = [0]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=6, line_per_file=1)
    
    shared_settings["--env_name"] = ["TTChangeAction/ConstPID"]
    shared_settings["--exp_info"] = ["/without_replay/action_-0.1_0.1/"]
    shared_settings["--action_scale"] = [0.2]
    shared_settings["--action_bias"] = [-0.1]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=12, line_per_file=1)
    
    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.1]
    shared_settings["--exp_info"] = ["/without_replay/change_0.1"]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [3]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=15, line_per_file=1)
    
    shared_settings["--env_name"] = ["TTChangeAction/DiscreteConstPID"]
    shared_settings["--env_info"] = [0.01]
    shared_settings["--exp_info"] = ["/without_replay/change_0.01"]
    shared_settings["--actor"] = ["Softmax"]
    shared_settings["--discrete_control"] = [1]
    shared_settings.pop('--action_scale', None)
    shared_settings.pop('action_bias', None)
    settings["GAC"]["--n"] = [3]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=18, line_per_file=1)


if __name__ == '__main__':
    # reproduce()
    # test_runs()
    # demo()
    constant_pid() # 26502
    # smpl_exp()
    # gem_exp()
