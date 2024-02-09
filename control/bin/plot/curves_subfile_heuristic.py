import copy
import itertools

from curves import DATAROOT, reproduce_demo, best_offline
from curves import visualize_training_info
from curves import sweep_parameter, draw_sensitivity, draw_sensitivity_2d
from utils import reduce_log_file_size


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

def c20240207():
    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/supervised_explore_wo_proposal/LineSearch/param_{}/"
    pths = {
        "scaler=1": [DATAROOT + SHAREPATH.format(0), "C0", 5],
        "scaler=5": [DATAROOT + SHAREPATH.format(1), "C1", 5],
        "scale=10": [DATAROOT + SHAREPATH.format(2), "C2", 5],
    }
    best_offline(pths, "linesearch_rho0.1", ylim=[-1, 1.2])
    pths = {
        "scaler=1": [DATAROOT + SHAREPATH.format(3), "C0", 5],
        "scaler=5": [DATAROOT + SHAREPATH.format(4), "C1", 5],
        "scale=10": [DATAROOT + SHAREPATH.format(5), "C2", 5],
    }
    best_offline(pths, "linesearch_rho0.2", ylim=[-1, 1.2])
    pths = {
        "scaler=1": [DATAROOT + SHAREPATH.format(6), "C0", 5],
        "scaler=5": [DATAROOT + SHAREPATH.format(7), "C1", 5],
        "scale=10": [DATAROOT + SHAREPATH.format(8), "C2", 5],
    }
    best_offline(pths, "linesearch_rho0.5", ylim=[-1, 1.2])

    pths = {
        "rho=0.1": [DATAROOT + SHAREPATH.format(0), "C0", 5],
        "rho=0.2": [DATAROOT + SHAREPATH.format(3), "C1", 5],
        "rho=0.5": [DATAROOT + SHAREPATH.format(6), "C2", 5],
    }
    best_offline(pths, "linesearch_scaler1", ylim=[-1, 1.2])
    pths = {
        "rho=0.1": [DATAROOT + SHAREPATH.format(1), "C0", 5],
        "rho=0.2": [DATAROOT + SHAREPATH.format(4), "C1", 5],
        "rho=0.5": [DATAROOT + SHAREPATH.format(7), "C2", 5],
    }
    best_offline(pths, "linesearch_scaler5", ylim=[-1, 1.2])
    pths = {
        "rho=0.1": [DATAROOT + SHAREPATH.format(2), "C0", 5],
        "rho=0.2": [DATAROOT + SHAREPATH.format(5), "C1", 5],
        "rho=0.5": [DATAROOT + SHAREPATH.format(8), "C2", 5],
    }
    best_offline(pths, "linesearch_scaler10", ylim=[-1, 1.2])

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/supervised_explore_wo_proposal/"
    fixed_params_list = {
        "lr_critic": [1],
    }
    draw_sensitivity_2d(DATAROOT + SHAREPATH, 'LineSearch', fixed_params_list, "rho", "exploration", "sensitivity_linesearch")


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
    # visualize_training_info(file, target_key, title="heuristic_batch", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_wo_resetting/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/LineSearchBU/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="heuristic_batch_wo_resetting", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/NonContexTT/heuristic_lr_wo_resetting/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1.1_clip_50/activation_relu/optim_sgd/LineSearchBU/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="heuristic_bound", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    # SHAREPATH = "output/test_v1/NonContexTT/heuristic_separate_testset/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.1_clip_20/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    SHAREPATH = "output/test_v1/NonContexTT/heuristic_separate_testset/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.1_clip_50/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="heuristic_sep_test", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    # SHAREPATH = "output/test_v1/NonContexTT/heuristic_separate_testset/setpoint_3/obs_raw/action_scale9.9_bias0.1/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.1_clip_50/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    SHAREPATH = "output/test_v1/NonContexTT/heuristic_separate_testset/setpoint_3/obs_raw/action_scale9.8_bias0.2/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.0_clip10000/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="heuristic_sep_test", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/obs_raw/action_scale5_bias-5/reward_clip[-1,1]/replay5000_batch512/beta_shift_1.0_clip10000/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="heuristic_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/rl_setting/batch_512/beta_shift1_clip10000/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/supervised_explore_network/1x_explore_bonus/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/supervised_explore_network/5x_explore_bonus/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/supervised_explore_network/10x_explore_bonus/LineSearch/param_0/seed_0"
    file = DATAROOT + SHAREPATH
    # visualize_training_info(file, target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration_use_critic_lr/setpoint_3/supervised_explore_wo_proposal/LineSearch/param_{}/seed_0"
    file = DATAROOT + SHAREPATH
    # for i in range(9):
    #     visualize_training_info(file.format(i), target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration/setpoint_3/supervised_explore/LineSearch/param_{}/seed_0"
    file = DATAROOT + SHAREPATH
    # for i in range(13):
    #     visualize_training_info(file.format(i), target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_exploration/setpoint_3/bootstrap_explore/LineSearch/param_{}/seed_0"
    file = DATAROOT + SHAREPATH
    # for i in range(4):
    #     visualize_training_info(file.format(i), target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_refactor/setpoint_3/bootstrap_explore/LineSearchGAC/param_{}/seed_0"
    file = DATAROOT + SHAREPATH
    # for i in range(4):
    #     visualize_training_info(file.format(i), target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_refactor/setpoint_3/supervised_explore/LineSearchGAC/param_{}/seed_0"
    file = DATAROOT + SHAREPATH
    # for i in range(4):
    #     visualize_training_info(file.format(i), target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

    SHAREPATH = "output/test_v1/TTChangeAction/ConstPID/heuristic_refactor/setpoint_3/bootstrap_from_random_explore/LineSearchGAC/param_{}/seed_0"
    file = DATAROOT + SHAREPATH
    for i in range(1, 4):
        visualize_training_info(file.format(i), target_key, title="rl_change_action", threshold=0.99, xlim=None, ylim=[-1, 1.2], log_scale_keys=log_scale_keys)

def clean_log_file():
    reduce_log_file_size(DATAROOT + "output/test_v1/NonContexTT/heuristic_separate_testset/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1_clip_20/activation_relu/optim_sgd/LineSearch")

if __name__ == '__main__':
    target_key = [
        "actor_info/param1",
        "proposal_info/param1",
        "actor_info/param2",
        "proposal_info/param2",
        "critic_info/Q",
        "explore_bonus",
        "LS_critic/lr",
        "LS_critic/lr_weight",
        "LS_actor/lr",
        "LS_actor/lr_weight",
        "LS_sampler/lr",
        "LS_sampler/lr_weight",
        "LS_explore/lr",
        "LS_explore/lr_weight",
        # "lr_actor",
        # "lr_critic",
        # "lr_actor_scaler",
        # "lr_sampler_scaler",
        # "lr_critic_scaler",
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
        "explore_bonus",
        "LS_critic/lr",
        "LS_critic/lr_weight",
        "LS_actor/lr",
        "LS_actor/lr_weight",
        "LS_sampler/lr",
        "LS_sampler/lr_weight",
        "LS_explore/lr",
        "LS_explore/lr_weight",
        # "lr_actor",
        # "lr_critic",
        # "lr_actor_scaler",
        # "lr_sampler_scaler",
        # "lr_critic_scaler",
        # "lr_explore_scaler",
    ]

    # c20240129_1()
    # c20240207()
    visualization()

    # clean_log_file()