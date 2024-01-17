import copy
import itertools

from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline
from curves import visualize_training_info, sensitivity_plot, sensitivity_plot_2d


def sweep_parameter(pth_base, agent_list=['GAC']):
    for agent in agent_list:
        sweep_offline(pth_base+"/{}/".format(agent), agent)


def draw_sensitivity(pth_base, agent, fix_params_list, sweep_param, title):
    keys, values = zip(*fix_params_list.items())
    fix_params_choices = [dict(zip(keys, v)) for v in itertools.product(*values)]
    sensitivity_plot(pth_base+"/{}/".format(agent), agent, fix_params_choices, sweep_param, title)

def draw_sensitivity_2d(pth_base, agent, fix_params_list, sweep_param1, sweep_param2, title):
    keys, values = zip(*fix_params_list.items())
    fix_params_choices = [dict(zip(keys, v)) for v in itertools.product(*values)]
    sensitivity_plot_2d(pth_base+"/{}/".format(agent), agent, fix_params_choices, sweep_param1, sweep_param2, title)

def visualize_temp():
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

    file = DATAROOT + "output/test_v0/TTAction/ConstPID/temp/GACPS/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_temp", threshold=0.99, xlim=None, ylim=[-2, 2])

def c20240116():
    SHAREPATH = "../out/output/test_v1/NonContexTT/small_network/setpoint_3/obs_raw/action_raw/replay1_batch1/beta_shift_0/"
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

    SHAREPATH = "../out/output/test_v1/NonContexTT/small_network/setpoint_3/obs_raw/action_scale/replay1_batch1/beta_shift_0/"
    sweep_parameter(DATAROOT + SHAREPATH, ['GAC'])
    pths = {
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_19", "C0", 5],
    }
    best_offline(pths, "directly_learn_beta_scale_action_setpoint3", ylim=[-2, 2])

if __name__ == '__main__':
    c20240116()