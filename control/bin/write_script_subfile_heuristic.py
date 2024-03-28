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

def c20240131(settings, shared_settings, target_agents):
    """
    Higher beta parameter bound (beta bias and upper bound)
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearchBU  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1.1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 256  --exp_name /heuristic_lr_wo_resetting_higher_bound/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 256  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1.1_clip_50/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 50 --max_step 5000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

    """
    Use a separated testset
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1.1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_separate_testset/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 256  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.1_clip_20/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 20 --max_step 5000",
            "python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 10.0  --action_bias 0.0  --beta_parameter_bias 1.1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_separate_testset/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 256  --exp_info /setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.1_clip_50/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 50 --max_step 5000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240201(settings, shared_settings, target_agents):
    """
    Clip the action to [0.1, 10]
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 9.9  --action_bias 0.1  --beta_parameter_bias 1.1  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_separate_testset/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 256  --exp_info /setpoint_3/obs_raw/action_scale9.9_bias0.1/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.1_clip_50/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 50 --max_step 5000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

    """    
    Change the beta parameter's bias back to 1
    Increase the beta parameter's upper bound to 10,000
    Clip the action at [0.2, 10]

    *** This is a stable agent config ***
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --env_action_scaler 1.0  --action_scale 9.8  --action_bias 0.2  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_separate_testset/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 512  --exp_info /setpoint_3/obs_raw/action_scale9.8_bias0.2/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.0_clip10000/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --max_step 5000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

    """
    Test the random network exploration
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name NonContexTT  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 512  --exp_info /setpoint_3/obs_raw/action_scale9.8_bias0.2/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.0_clip10000/activation_relu/optim_sgd/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --max_step 5000"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240202(settings, shared_settings, target_agents):
    """
    Commit 1c7100a4dd4ba0b9ddef624728f3ebe74e15622b
    Add safty setting in changed action pid, reset the pid parameter to [1.2, 15] when the reward is less than -1
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /rl_setting/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /batch_512/beta_shift1_clip10000  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9"]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240205(settings, shared_settings, target_agents):
    # """
    # Test in Acrobot
    # """
    # cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 0  --render 0  --polyak 0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --env_info 1.0 3.0  --discrete_control 1 --buffer_size 50000  --batch_size 512  --exp_name /test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /batch_512/softmax/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --gamma 0.95"]
    # write_cmd(cmds, prev_file=0, line_per_file=1)

    """
    Test supervised exploration network learning
    """
    cmds = ["python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore_network/1x_explore_bonus/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9",]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240207(settings, shared_settings, target_agents):
    """
    Back track the exploration network learning rate
    commit 7519230560815ebcdc7fdf81d39391b02080d818
    param 0-8: without proposal policy
    param 9-11: with proposal policy
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 1  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 2  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",
        "python3 main.py  --param 3  --agent_name LineSearch  --rho 0.2  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 4  --agent_name LineSearch  --rho 0.2  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 5  --agent_name LineSearch  --rho 0.2  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",
        "python3 main.py  --param 6  --agent_name LineSearch  --rho 0.5  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 7  --agent_name LineSearch  --rho 0.5  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 8  --agent_name LineSearch  --rho 0.5  --tau 0  --prop_rho_mult 1  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",

        "python3 main.py  --param 9  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 10  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 11  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",

        "python3 main.py  --param 12  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 0",
    ]
    # write_cmd(cmds, prev_file=0, line_per_file=1)

    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 1  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 2  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",

        "python3 main.py  --param 3  --agent_name LineSearch  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_exploration/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 0",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240208(settings, shared_settings, target_agents):
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 0",
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 3  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",
    ]
    # write_cmd(cmds, prev_file=0, line_per_file=1)

    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 0",
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 3  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/supervised_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240209(settings, shared_settings, target_agents):
    """
    Follow random policy in bootstrap exploration
    """
    cmds = [
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 5",
        "python3 main.py  --param 3  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",
    ]
    # write_cmd(cmds, prev_file=0, line_per_file=1)
    """
    Test exploration in acrobot
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 50000 --timeout 1000 --log_interval 500 --stats_queue_size 5 --debug 0  --render 0  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --env_info 1.0 3.0  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /heuristic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /bootstrap_from_random_explore/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 0",
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 50000 --timeout 1000 --log_interval 500 --stats_queue_size 5 --debug 0  --render 0  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --env_info 1.0 3.0  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /heuristic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /bootstrap_from_random_explore/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 50000 --timeout 1000 --log_interval 500 --stats_queue_size 5 --debug 0  --render 0  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --env_info 1.0 3.0  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /heuristic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /bootstrap_from_random_explore/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 10",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240212(settings, shared_settings, target_agents):
    """
    check
    change sgd to rmsprop
    in changed action, run longer
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_bonus/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 20000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_bonus/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer RMSprop  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_bonus/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240214(settings, shared_settings, target_agents):
    # """
    # Test CQLGAC for offline learning
    # """
    # cmds = [
    #     "python3 main.py  --param 0  --agent_name CQLGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 5000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_bonus/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_from_random_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1",
    # ]
    # write_cmd(cmds, prev_file=0, line_per_file=1)

    """
    Test LineSearchGAC with two Q network

    Commit 66cc181099b5f411b43647c4bac9323e24679dde
    Added ensemble critic
    Save log to test_v2
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_0/SGD/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_0/SGD/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_0/Adam/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_0/Adam/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_0/SGD/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_0/SGD/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_0/Adam/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_0/Adam/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1",

        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_100/SGD/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_100/SGD/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_100/Adam/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/bootstrap_from_random_explore/prefill_100/Adam/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_100/SGD/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_100/SGD/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_100/Adam/rho0.1/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --buffer_prefill 100",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 200  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /ensemble_critic/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/bootstrap_from_random_explore/prefill_100/Adam/rho0.4/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --buffer_prefill 100",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240215(settings, shared_settings, target_agents):
    """
    Test policy resetting in LineSearchGAC
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchReset  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --reset_mode None --reset_param 0",
        "python3 main.py  --param 1  --agent_name LineSearchReset  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --reset_mode Random --reset_param 0",
        "python3 main.py  --param 2  --agent_name LineSearchReset  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --reset_mode Shift --reset_param=-1.",
        "python3 main.py  --param 3  --agent_name LineSearchReset  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --reset_mode Shrink --reset_param 0.5",
        "python3 main.py  --param 4  --agent_name LineSearchReset  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2 --reset_mode Shrink+Rnd --reset_param 0.5",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240216(settings, shared_settings, target_agents):
    """
    Fix linesearch with Adam and RMSprop optimizers
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /temp/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/Adam/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.4  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer RMSprop  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /temp/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_2/RMSprop/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 2",
    ]
    # write_cmd(cmds, prev_file=0, line_per_file=1)

    """
    Target network v.s. min Q
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 50000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /SGD/target_0.995/ensemble1/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1 --critic_ensemble 1  --n 30",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 50000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /SGD/target_0.995/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1 --critic_ensemble 2  --n 30",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 50000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /SGD/target_0/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1 --critic_ensemble 2  --n 30",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 50000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble1/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1 --critic_ensemble 1  --n 30",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 50000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1 --critic_ensemble 2  --n 30",
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 50000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 50000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.95 --exploration 1 --critic_ensemble 2  --n 30",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240220(settings, shared_settings, target_agents):
    """
    Sweep Adam optimizer
    """
    cmds = [
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble1/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 10 --critic_ensemble 1  --n 30  --optimizer_param 0.9 0.999 1e-08",
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 10 --critic_ensemble 2  --n 30  --optimizer_param 0.9 0.999 1e-08",
        "python3 main.py  --param 1  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 10 --critic_ensemble 2  --n 30  --optimizer_param 0.9 0.999 1e-08",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble1/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 10 --critic_ensemble 1  --n 30  --optimizer_param 0.95 0.95 1e-04",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 10 --critic_ensemble 2  --n 30  --optimizer_param 0.95 0.95 1e-04",
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /compare_minQ_targetNet/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0/ensemble2/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 10 --critic_ensemble 2  --n 30  --optimizer_param 0.95 0.95 1e-04",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240221(settings, shared_settings, target_agents):
    """
    Test policy resetting in LineSearchGAC
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchReset  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --reset_mode None",
        "python3 main.py  --param 1  --agent_name LineSearchReset  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --reset_mode Random --reset_param 0",
        "python3 main.py  --param 2  --agent_name LineSearchReset  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --reset_mode Shift --reset_param=-1.",
        "python3 main.py  --param 3  --agent_name LineSearchReset  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --reset_mode Shrink --reset_param 0.5",
        "python3 main.py  --param 4  --agent_name LineSearchReset  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 2  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /reset_test/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /setpoint_3/ensemble_1/SGD/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 1 --critic_ensemble 1 --reset_mode Shrink+Rnd --reset_param 0.5",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240222(settings, shared_settings, target_agents):
    """
    Check Customized Adam optimizer (X)
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 500 --log_interval 500 --stats_queue_size 1 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /temp/  --activation ReLU  --lr_actor 1  --lr_critic 1  --exp_info /Adam/target_0.995/ensemble1/  --seed 0 --actor Softmax --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 1 --critic_ensemble 1  --n 30  --optimizer_param 0.9 0.999 1e-08",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240226(settings, shared_settings, target_agents):
    """
    A working cmd for GAC in Acrobot, for debugging.
        - Adam
        - Customized Adam
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name GAC  --rho 0.1  --tau 1e-2  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 32  --exp_name /temp/  --activation ReLU  --lr_actor 1e-3  --lr_critic 1e-3  --exp_info /temp/  --seed 0 --actor Softmax --hidden_actor 64 64 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 1  --n 30  --optimizer_param 0.9 0.999 1e-08 --buffer_prefill 10000",
        "python3 main.py  --param 0  --agent_name GAC  --rho 0.1  --tau 1e-2  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer CustomAdam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 32  --exp_name /temp/  --activation ReLU  --lr_actor 1e-3  --lr_critic 1e-3  --exp_info /temp/  --seed 0 --actor Softmax --hidden_actor 64 64 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 1  --n 30  --optimizer_param 0.9 0.999 1e-08 --buffer_prefill 10000",
    ]
    """
    A test cmd for LineSearch+Adam in Acrobot, for debugging.
        - Customized Adam
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 1e-2  --prop_rho_mult 5  --version 2  --max_steps 300000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer CustomAdam  --env_name Acrobot-v1  --discrete_control 1  --buffer_size 300000  --batch_size 32  --exp_name /temp/  --activation ReLU  --lr_actor 1e-3  --lr_critic 1e-2  --exp_info /temp/  --seed 0 --actor Softmax --hidden_actor 64 64 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 2  --optimizer_param 0.9 0.999 1e-08"
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

def c20240228(settings, shared_settings, target_agents):
    """
    A working cmd for linesearch+CustomAdam in Pendulum
    """
    cmds = [
        "python3 main.py  --param 0  --agent_name LineSearchGAC  --rho 0.1  --tau 1e-2  --prop_rho_mult 5  --version 2  --max_steps 100000 --timeout 1000 --log_interval 1000 --stats_queue_size 5 --debug 0  --render 0  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer CustomAdam  --env_name Pendulum-v1  --discrete_control 1  --buffer_size 300000  --batch_size 512  --exp_name /temp/  --activation ReLU  --lr_actor 1e-3  --lr_critic 1e-3  --exp_info /temp/  --seed 0 --actor Softmax --hidden_actor 64 64  --hidden_critic 64 64 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 1 --critic_ensemble 1  --optimizer_param 0.9 0.999 1e-08 --buffer_prefill 0  --n 30 --max_backtracking 30"
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

if __name__=='__main__':
    settings = {
        "LineSearchGAC": {
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
    target_agents = ["LineSearchGAC"]

    # c20240220(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
    c20240221(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
