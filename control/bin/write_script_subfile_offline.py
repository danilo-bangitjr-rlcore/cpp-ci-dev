import copy

from write_script import combinations, merge_independent, write_cmd
"""
Initial test on offline agent in d4rl
"""

def c20240228(settings, shared_settings, target_agents):
    cmds = [
        "python3 main.py  --param 0  --agent_name IQL  --rho 0.7  --tau 3  --version 2  --max_steps 1000000 --timeout 1000 --log_interval 10000 --log_test 1 --stats_queue_size 5  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Walker2d-expert  --discrete_control 0  --buffer_size 1000000  --batch_size 256  --exp_name /offline_test/  --activation ReLU  --lr_actor 3e-4  --lr_critic 3e-4  --exp_info /test0/  --seed 0 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 2  --buffer_prefill 0",
        "python3 main.py  --param 0  --agent_name IQL  --rho 0.7  --tau 3  --version 2  --max_steps 1000000 --timeout 1000 --log_interval 10000 --log_test 1 --stats_queue_size 5  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Walker2d-medium  --discrete_control 0  --buffer_size 1000000  --batch_size 256  --exp_name /offline_test/  --activation ReLU  --lr_actor 3e-4  --lr_critic 3e-4  --exp_info /test0/  --seed 0 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 2  --buffer_prefill 0",
        "python3 main.py  --param 0  --agent_name InAC  --tau 0.01  --version 2  --max_steps 1000000 --timeout 1000 --log_interval 10000 --log_test 1 --stats_queue_size 5  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Walker2d-expert  --discrete_control 0  --buffer_size 1000000  --batch_size 256  --exp_name /offline_test/  --activation ReLU  --lr_actor 3e-4  --lr_critic 3e-4  --exp_info /test0/  --seed 0 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 2  --buffer_prefill 0",
        "python3 main.py  --param 0  --agent_name InAC  --tau 0.1  --version 2  --max_steps 1000000 --timeout 1000 --log_interval 10000 --log_test 1 --stats_queue_size 5  --use_target_network 1  --polyak 0.995  --beta_parameter_bias 1.0  --head_activation ReLU  --optimizer Adam  --env_name Walker2d-medium  --discrete_control 0  --buffer_size 1000000  --batch_size 256  --exp_name /offline_test/  --activation ReLU  --lr_actor 3e-4  --lr_critic 3e-4  --exp_info /test0/  --seed 0 --layer_init_actor Xavier/1 --beta_parameter_bound 10000 --gamma 0.99 --exploration 0 --critic_ensemble 2  --buffer_prefill 0",
    ]
    write_cmd(cmds, prev_file=0, line_per_file=1)

if __name__=='__main__':
    settings = {
    }
    shared_settings = {
    }
    target_agents = ["IQL"]

    c20240228(copy.deepcopy(settings), copy.deepcopy(shared_settings), copy.deepcopy(target_agents))
