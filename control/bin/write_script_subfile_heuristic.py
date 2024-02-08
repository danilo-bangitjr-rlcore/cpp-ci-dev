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
        "python3 main.py  --param 2  --agent_name LineSearchGAC  --rho 0.1  --tau 0  --prop_rho_mult 2  --version 1  --max_steps 1000  --debug 1  --render 2  --polyak 0  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer SGD  --action_normalizer Scale  --env_name TTChangeAction/ConstPID  --env_info 1.0 3.0  --buffer_size 10000  --batch_size 512  --exp_name /heuristic_refactor/  --activation ReLU  --lr_actor 1  --lr_critic 1  --etc_buffer_prefill 1  --exp_info /setpoint_3/bootstrap_explore/  --seed 0 --actor Beta --hidden_actor 256 256 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.9 --exploration 10",
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

    c20240208(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
