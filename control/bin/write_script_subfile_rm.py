import copy

from write_script import combinations, merge_independent

def learning_rate_sweep_adam_RMSprop(settings, shared_settings, target_agents):
    """
    Sweeps over learning rates for adam and RMSprop
    
    Ran on Jan 9, 2023
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
    Compares prefilling the buffer to now prefilling
    
    Ran on Jan 9, 2023
    """
    settings = {
        "GAC": {
        },
    }
    
    # uncomment these, to generate two runfiles, then merge them manually
    shared_settings = {
        "--exp_name": ["buffer_prefill"],
        "--max_steps": [5000],
        "--render": [2],
        "--env_action_scaler": [10],
        "--action_scale": [1],
        "--action_bias": [0],
        "--optimizer" : ['RMSprop'],
        "--tau": [2, 1, 1e-1],
        "--rho": [0.2],
        "--lr_actor": [1, 0.5, 0.25, 0.1,],
        "--lr_critic": [0.01, 0.001, 0.0001, 0.00001]
    }
    
    # shared_settings = {
    #     "--exp_name": ["buffer_prefill"],
    #     "--max_steps": [5000],
    #     "--render": [2],
    #     "--env_action_scaler": [10],
    #     "--action_scale": [1],
    #     "--action_bias": [0],
    #     "--optimizer" : ['RMSprop'],
    #     "--tau": [0],
    #     "--rho": [0.2],
    #     "--theta": [0.2, 0.4, 0.8],
    #     "--lr_actor": [1, 0.5, 0.25, 0.1,],
    #     "--lr_critic": [0.01, 0.001, 0.0001, 0.00001]
    # }
    
    target_agents = ["GAC"]
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["buffer_prefill"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [8, 32]
    shared_settings["--buffer_prefill"] = [100, 1000]
    shared_settings["--debug"] = [1]
    settings = merge_independent(settings, shared_settings)

    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=10000, comb_num_base=0)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=10000, comb_num_base=0)


def etc_critic_prefill(settings, shared_settings, target_agents):
    """
    Prefill the buffer, then start learning
    
    Ran on Jan 10, 2023
    """
    settings = {
        "ETC": {
        },
    }
    shared_settings = {
        "--exp_name": ["etc_critic/"],
        "--max_steps": [100000],
        "--render": [2],
        "--optimizer" : ['RMSprop'],
        "--lr_critic": [10**i for i in range(-2, -6, -1)],
        "--etc_learning_start": [2500],
        "--debug" : [1]
    }
    target_agents = ["ETC"]

    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["etc_critic"]
    shared_settings["--buffer_size"] = [5000]
    shared_settings["--batch_size"] = [64]
    shared_settings["--etc_buffer_prefill"] = [2500]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1000, comb_num_base=0)
    
def etc_critic_online(settings, shared_settings, target_agents):
    """
    start learning from the start
    
    Ran on Jan 10, 2023
    """
    settings = {
        "ETC": {
        },
    }
    shared_settings = {
        "--exp_name": ["etc_critic"],
        "--max_steps": [20000],
        "--render": [2],
        "--optimizer" : ['RMSprop'],
        "--lr_critic": [10**i for i in range(-2, -6, -1)],
        "--etc_learning_start": [0],
        "--debug" : [1]
    }
    target_agents = ["ETC"]

    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--exp_info"] = ["etc_critic/"]
    shared_settings["--buffer_size"] = [1000]
    shared_settings["--batch_size"] = [8, 64, 128]
    shared_settings["--etc_buffer_prefill"] = [0000]
    
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=4, line_per_file=1000, comb_num_base=16)
    
    
def GAC_Pendulum(settings, shared_settings, target_agents):
    """
    Runs the same experiment from the original GAC paper with our agent
    https://arxiv.org/pdf/1810.09103.pdf
    """
    settings = {
        "GAC": {
        },
    }
    shared_settings = {
        "--exp_name": ["GAC_classic_control_and_PID"],
        "--max_steps": [20000],
        "--render": [0],
        "--auto_calibrate_beta_support" : [1],
        "--optimizer" : ["Adam"],
        "--tau": [10],
        "--rho": [0.1],
        "--lr_actor": [1e-1*1e-2],
        "--timeout" : [200],
        "--discrete_control": [0],
        "--actor": ['Beta'],
        "--lr_critic": [1e-2 ]    
    }
    
    target_agents = ["GAC"] 
    shared_settings["--env_name"] = ["Pendulum-v1"]
    shared_settings["--exp_info"] = ["GAC_classic_control/"]
    shared_settings["--buffer_size"] = [100000]
    shared_settings["--batch_size"] = [32]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=10, prev_file=0, line_per_file=1000, comb_num_base=0)
    
    
# def GAC_MountainCar(settings, shared_settings, target_agents):
#     """
#     Runs the same experiment from the original GAC paper with our agent
#     https://arxiv.org/pdf/1810.09103.pdf
#     """
#     settings = {
#         "GAC": {
#         },
#     }
#     shared_settings = {
#         "--exp_name": ["GAC_classic_control_and_PID"],
#         "--max_steps": [20000],
#         "--render": [0],
#         "--auto_calibrate_beta_support" : [1],
#         "--optimizer" : ["Adam"],
#         "--tau": [10],
#         "--rho": [0.1],
#         "--lr_actor": [1*1e-3],
#         # "--hidden_critic": [[64, 64]],
#         # "--hidden_actor": [[64, 64]],
#         "--timeout" : [200],
#         "--discrete_control": [0],
#         "--actor": ['Beta'],
#         "--lr_critic": [1e-3 ]    
#     }
    
#     target_agents = ["GAC"] 
#     shared_settings["--env_name"] = ["MountainCarContinuous-v0"]
#     shared_settings["--exp_info"] = ["GAC_classic_control/"]
#     shared_settings["--buffer_size"] = [100000]
#     shared_settings["--batch_size"] = [32]
#     settings = merge_independent(settings, shared_settings)
#     combinations(settings, target_agents, num_runs=10, prev_file=0, line_per_file=1000, comb_num_base=0)
    
    
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
    target_agents = ["SimpleAC", "SAC", "GAC", "ETC"]

    # learning_rate_sweep_adam_RMSprop(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # buffer_prefill(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # etc_critic(settings, shared_settings, target_agents)
    etc_critic_prefill(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # etc_critic_online(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    # GAC_Pendulum(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))