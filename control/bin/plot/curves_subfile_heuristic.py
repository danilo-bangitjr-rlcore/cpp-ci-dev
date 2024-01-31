import copy
import itertools

from curves import DATAROOT, reproduce_demo, best_offline
from curves import visualize_training_info
from curves import sweep_parameter, draw_sensitivity, draw_sensitivity_2d


def c20240129_1():
    SHAREPATH = "output/test_v1/NonContexTT/exp_betaBound_and_prefill/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/activation_relu/optim_sgd/"
    pths = {
        "Param0": [DATAROOT + SHAREPATH + "LineSearch/param_0", "C0", 5],
        # "Param3": [DATAROOT + SHAREPATH + "LineSearch/param_3", "C3", 3],
        "Param4": [DATAROOT + SHAREPATH + "LineSearch/param_4", "C3", 3],
    }
    best_offline(pths, "linesearch", ylim=[-1, 1.2])

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_exp/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1/activation_relu/optim_sgd/"
    pths = {
        "Param0": [DATAROOT + SHAREPATH + "LineSearch/param_0", "C0", 5],
    }
    best_offline(pths, "linesearch1", ylim=[-1, 1.2])

    fixed_params_list = {
        "tau": [0],
    }
    # draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "lr_actor", "lr_critic", "sensitivity_lr_recreate_obs0")


def visualization():

    SHAREPATH = "output/test_v1/NonContexTT/exp_betaBound_and_prefill/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/activation_relu/optim_sgd/LineSearch/param_2/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="sanity_check", threshold=0.99, xlim=None, ylim=[-1, 1.2])

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_exp/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/LineSearch/param_1/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="beta_shift_clip", threshold=0.99, xlim=None, ylim=[-1, 1.2])

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_exp/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="beta_shift", threshold=0.99, xlim=None, ylim=[-1, 1.2])

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_exp/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/LineSearchBU/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    visualize_training_info(file, target_key, title="heuristic_batch", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_wo_resetting/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/LineSearchBU/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    visualize_training_info(file, target_key, title="heuristic_batch_wo_resetting", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)


if __name__ == '__main__':
    target_key = [
        "actor_info/param1",
        "proposal_info/param1",
        "actor_info/param2",
        "proposal_info/param2",
        "critic_info/Q",
        "lr_actor",
        "lr_critic",
        "lr_actor_scaler",
        "lr_sampler_scaler",
        "lr_critic_scaler",
        "env_info/constrain_detail/kp1",
        "env_info/constrain_detail/tau",
        "env_info/constrain_detail/height",
        "env_info/constrain_detail/flowrate",
        # "env_info/constrain_detail/C1",
        # "env_info/constrain_detail/C2",
        # "env_info/constrain_detail/C3",
        # "env_info/constrain_detail/C4",
        # "env_info/lambda",
    ]
    log_scale_keys = [
        "lr_actor",
        "lr_critic",
        "lr_actor_scaler",
        "lr_sampler_scaler",
        "lr_critic_scaler",
    ]

    # c20240129_1()
    visualization()