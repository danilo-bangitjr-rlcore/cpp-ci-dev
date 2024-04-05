import copy
import itertools

from curves import DATAROOT, reproduce_demo, best_offline
from curves import visualize_training_info
from curves import sweep_parameter, draw_sensitivity, draw_sensitivity_2d


def demo():
    """
    ThreeTanks
    """
    pths = {
        "Baseline":[DATAROOT + "baseline/reproduce_new_reward_no_smooth/param_0/", "C0", 2],
    }
    reproduce_demo(pths, "reproduce", ylim=[-500, 2], xlim=[0, 5001])
    pths = {
        "Baseline":[DATAROOT + "baseline/reproduce_new_reward_no_smooth/param_0/", "C0", 2],
        "New":[DATAROOT + "output/test_v1/NonContexTT/recreating_results_vis/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU/GAC/param_0", "C1", 3]
    }
    reproduce_demo(pths, "demo_compare", ylim=[-200, 10], xlim=[0, 5001])
    # reproduce_demo(pths, "demo_compare_zoomin", ylim=[-50, 2])

    best_offline(pths, "demo_no_smooth", ylim=[-200, 10], xlim=[0, 5001])


def c20240116():
    SHAREPATH = "output/test_v1/NonContexTT/small_network/setpoint_3/obs_raw/action_raw/replay1_batch1/beta_shift_0/"
    # sweep_parameter(DATAROOT + SHAREPATH, ['GAC'])
    fixed_params_list = {
        "rho": [0.1],
        "tau": [1e-3],
    }
    # draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "lr_actor", "lr_critic", "sensitivity_lr_setpoint2-4")
    pths = {
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_19", "C0", 5],
    }
    # best_offline(pths, "directly_learn_beta_setpoint3", ylim=[-2, 2])

    SHAREPATH = "output/test_v1/NonContexTT/small_network/setpoint_3/obs_raw/action_scale/replay1_batch1/beta_shift_0/"
    sweep_parameter(DATAROOT + SHAREPATH, ['GAC'])
    pths = {
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_19", "C0", 5],
    }
    best_offline(pths, "directly_learn_beta_scale_action_setpoint3", ylim=[-2, 2])

def c20240117_0():
    # SHAREPATH = "output/test_v1/NonContexTT/recreating_results_vis/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/"
    # pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C0", 5],}
    # best_offline(pths, "recreate", ylim=[-2, 2])
    # file = DATAROOT + SHAREPATH + "/GAC/param_0/seed_0"
    # visualize_training_info(file, target_key, title="vis_recreate", threshold=0.995, xlim=None, ylim=[-2, 2])

    SHAREPATH = "output/test_v1/NonContexTT/recreating_results_vis/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU/"
    pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C0", 5],}
    best_offline(pths, "recreate_relu", ylim=[-2, 2])
    file = DATAROOT + SHAREPATH + "/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_recreate_relu", threshold=0.995, xlim=None, ylim=[-2, 2])

    SHAREPATH = "output/test_v1/NonContexTT/recreating_results_vis/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU/change_to_Adam/"
    pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C0", 5],}
    best_offline(pths, "recreate_relu_adam", ylim=[-2, 2])
    file = DATAROOT + SHAREPATH + "/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_recreate_relu_adam", threshold=0.995, xlim=None, ylim=[-2, 2])

    SHAREPATH = "output/test_v1/NonContexTT/recreating_results_vis/directly_learn_beta/totally_changed_envActionScaler1_betaScaler10_withActionNormalizer/change_to_ReLU/remove_action_normalizer/"
    pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C0", 5],}
    best_offline(pths, "recreate_relu_woAScale", ylim=[-2, 2])
    file = DATAROOT + SHAREPATH + "/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_recreate_relu_woAScale", threshold=0.995, xlim=None, ylim=[-2, 2])

def c20240118():
    SHAREPATH = "output/test_v1/NonContexTT/recreating_results_vis/nonlinear_beta/recreate_nonlinear_test/"
    pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C0", 5],}
    # best_offline(pths, "recreate_nonlinear", ylim=[-2, 2])

def c20240119():
    SHAREPATH = "../out/output/test_v1/NonContexTT/recreating_results_vis/sweep_obs0/"
    sweep_parameter(DATAROOT + SHAREPATH, ['GAC'])
    pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_73", "C0", 5],}
    best_offline(pths, "recreate_nonlinear_obs0", ylim=[-2, 2])
    fixed_params_list = {
        "tau": [0],
    }
    draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "lr_actor", "lr_critic", "sensitivity_lr_recreate_obs0")

    SHAREPATH = "output/test_v1/NonContexTT/recreating_results_vis/nonlinear_beta/sweep_obs1/"
    sweep_parameter(DATAROOT + SHAREPATH, ['GAC'])
    pths = {"GAC": [DATAROOT + SHAREPATH + "GAC/param_52", "C0", 5],}
    best_offline(pths, "recreate_nonlinear_obs1", ylim=[-2, 2])
    fixed_params_list = {
        "tau": [0],
    }
    draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "lr_actor", "lr_critic", "sensitivity_lr_recreate_obs1")

def visualization():

    target_key = [
        "actor_info/param1",
        "proposal_info/param1",
        "actor_info/param2",
        "proposal_info/param2",
        "critic_info/Q",
        "env_info/constrain_detail/kp1",
        "env_info/constrain_detail/tau",
        "env_info/constrain_detail/height",
        "env_info/constrain_detail/flowrate",
        "env_info/constrain_detail/C1",
        "env_info/constrain_detail/C2",
        "env_info/constrain_detail/C3",
        "env_info/constrain_detail/C4",
        # "env_info/constrain",
        "env_info/lambda",
    ]

    file = DATAROOT + "output/test_v1/NonContexTT/heuristic_lr_exp/setpoint_3/obs_raw/action_scale/reward_clip[-1,1]/replay5000_batch256/beta_shift_1/activation_relu/optim_sgd/LineSearch/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_temp", threshold=0.99, xlim=None, ylim=[-2, 2])

if __name__ == '__main__':
    target_key = [
        "actor_info/param1",
        "proposal_info/param1",
        "actor_info/param2",
        "proposal_info/param2",
        "critic_info/Q",
        "env_info/constrain_detail/kp1",
        "env_info/constrain_detail/tau",
        "env_info/constrain_detail/height",
        "env_info/constrain_detail/flowrate",
        "env_info/constrain_detail/C1",
        "env_info/constrain_detail/C2",
        "env_info/constrain_detail/C3",
        "env_info/constrain_detail/C4",
        # "env_info/constrain",
        "env_info/lambda",
    ]

    # demo()
    # c20240118()
    # c20240119()
    visualization()