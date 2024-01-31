import copy

from write_script import combinations, merge_independent, write_cmd
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
    # shared_settings["--layer_init_actor"] = ["Const/10/0"]
    shared_settings["--exp_name"] = ["heuristic_lr_exp"]
    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--reward_normalizer"] = ["Clip/-1/1"]
    shared_settings["--activation"] = ["ReLU"]
    shared_settings["--lr_actor"] = [1.0]
    shared_settings["--lr_critic"] = [1.0]
    shared_settings["--etc_buffer_prefill"] = [5000]
    shared_settings["--debug"] = [1]
    shared_settings["--render"] = [2]
    shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/replay5000_batch256/beta_shift_1/activation_relu"]

    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)
    """
    python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1 --prop_rho_mult 2 --tau 0  --version 1  --max_steps 5000  --debug 0  --render 0  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 5000  --batch_size 256  --exp_name heuristic_lr_exp  --reward_normalizer Clip/-1/1  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 5000  --exp_info /setpoint_3/obs_raw/action_scale/replay5000_batch256/beta_shift_1/activation_relu  --seed 0
    """

def c20240129_0(settings, shared_settings, target_agents):
    """
    Test runs
    """
    """
        Go back to linear policy network
        Use SGD
        Clip the reward from -1 to 1
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 5000  --batch_size 256  --exp_name heuristic_lr_exp  --reward_normalizer Clip/-1/1  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 0  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 0 --layer_init_actor Const/10/0"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

    """
        Nonlinear policy network
        Use SGD
        Clip the reward from -1 to 1
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 1  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 256  --exp_name heuristic_lr_exp  --reward_normalizer Clip/-1/1  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 0  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 20"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240129_1(settings, shared_settings, target_agents):
    shared_settings["--env_name"] = ["NonContexTT"]
    shared_settings["--env_info"] = [[1., 3.]]
    shared_settings["--buffer_size"] = [10000]
    shared_settings["--batch_size"] = [256]
    shared_settings["--action_normalizer"] = ["Scale"]
    shared_settings["--activation"] = ["ReLU"]
    shared_settings["--optimizer"] = ["SGD"]
    shared_settings["--lr_actor"] = [1.0]
    shared_settings["--lr_critic"] = [1.0]
    shared_settings["--debug"] = [1]
    shared_settings["--render"] = [2]

    shared_settings["--etc_buffer_prefill"] = [0, 5000]
    shared_settings["--beta_parameter_bound"] = [20, 50, 100, 1000]
    shared_settings["--exp_name"] = ["exp_betaBound_and_prefill"]
    shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/activation_relu/optim_sgd/"]
    settings = merge_independent(settings, shared_settings)
    combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

def c20240130(settings, shared_settings, target_agents):
    # target_agents = ["LineSearchBU"]
    #
    # shared_settings["--env_name"] = ["NonContexTT"]
    # shared_settings["--env_info"] = [[1., 3.]]
    # shared_settings["--buffer_size"] = [10000]
    # shared_settings["--batch_size"] = [256]
    # shared_settings["--action_normalizer"] = ["Scale"]
    # shared_settings["--activation"] = ["ReLU"]
    # shared_settings["--optimizer"] = ["SGD"]
    # shared_settings["--lr_actor"] = [1.0]
    # shared_settings["--lr_critic"] = [1.0]
    # shared_settings["--debug"] = [1]
    # shared_settings["--render"] = [2]
    #
    # shared_settings["--etc_buffer_prefill"] = [0, 5000]
    # shared_settings["--beta_parameter_bound"] = [20, 50, 0]
    # shared_settings["--exp_name"] = ["exp_betaBound_and_prefill"]
    # shared_settings["--exp_info"] = ["/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/activation_relu/optim_sgd/"]
    # settings = merge_independent(settings, shared_settings)
    # combinations(settings, target_agents, num_runs=1, prev_file=0, line_per_file=1, comb_num_base=0)

    """
    Without actor resetting
    With prefilling, batch update
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearchBU  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 256  --exp_name /heuristic_lr_exp/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 10000  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 20 --max_step 1000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

    """
    Without critic and actor resetting
    Remove prefilling, batch update
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearchBU  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 256  --exp_name /heuristic_lr_wo_resetting/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 256  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 20 --max_step 1000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)


if __name__=='__main__':
    settings = {
        "LineSearch": {
            "--rho": [0.1],
            "--tau": [0],
            "--prop_rho_mult": [2],
        },
        "LineSearchBU": {
            "--rho": [0.1],
            "--tau": [0],
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
        "--action_scale": [10.],
        "--action_bias": [0.],
        "--beta_parameter_bias": [1.],

        "--head_activation": ["ReLU"],
        "--optimizer": ["Adam"],
        "--action_normalizer": ["Scale"],
    }
    target_agents = ["LineSearch"]

    c20240130(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
