import itertools

from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline, sensitivity_plot_2d
from curves import visualize_training_info


def draw_sensitivity_2d():
    fixed_params_list = {
        "optimizer": ['Adam', 'RMSprop']
    }
    
    pth_base = DATAROOT + "output/test_v0/NonContexTT/learning_rate_sweep_adam_RMS_prop/target0/replay5000_batch8/env_scale_10/"
    keys, values = zip(*fixed_params_list.items())
    fix_params_choices = [dict(zip(keys, v)) for v in itertools.product(*values)]
   
    sensitivity_plot_2d(pth_base+"/{}/".format('GAC'), 'GAC', fix_params_choices, 'lr_actor', 'lr_critic', 'Sensitivity')
    
    
def compare_algorithms():
    """
    Reactor Environment. Compare GAC, SAC, and REINFORCE
    SHAREPATH = "output/test_v0/ReactorEnv/With_LR_Param_Baseline/Param_Sweep/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_20/", "C0", 5],
        "Reinforce": [DATAROOT + SHAREPATH + "Reinforce/param_14/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_91", "C1", 1],
    }
    best_offline(pths, "ReactorEnv_Baseline", ylim=[-1500, 0])

    SHAREPATH = "output/test_v0/Cont-CC-PermExDc-v0/With_LR_Param_Baseline/Episodic/Param_Sweep/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_37/", "C0", 5],
        "Reinforce": [DATAROOT + SHAREPATH + "Reinforce/param_0/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_114", "C1", 1],
    }
    best_offline(pths, "Cont-CC-PermExDc-v0_Baseline", ylim=[-50, 0])
    """
    SHAREPATH = "output/test_v0/NonContexTT/Noncontext_PID_Baseline/"
    pths = {
        "ETC": [DATAROOT + SHAREPATH + "ETC/param_0/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_28", "C1", 1],
    }
    best_offline(pths, "PID_Baseline", ylim=[-10, 2])

def test():
    file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/temp/GAC/param_0/seed_0"
    target_key = [
        "agent_info/param1",
        "agent_info/param2",
        # "env_info/constrain_detail/kp1",
        # "env_info/constrain_detail/tau",
        "env_info/constrain_detail/height",
        "env_info/constrain_detail/flowrate",
        # "env_info/constrain_detail/C1",
        # "env_info/constrain_detail/C2",
        "env_info/constrain_detail/C3",
        "env_info/constrain_detail/C4",
        # "env_info/constrain",
        "env_info/lambda",
    ]
    visualize_training_info(file, target_key, threshold=-10)


if __name__ == '__main__':
    draw_sensitivity_2d()
    # sweep_parameter()
    # compare_algorithms()


