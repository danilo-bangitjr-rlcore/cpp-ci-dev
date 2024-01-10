from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline
from curves import visualize_training_info


def sweep_parameter():
    # pth_base = DATAROOT + "output/test_v0/ReactorEnv/With_LR_Param_Baseline/Param_Sweep/{}/"
    # pth_base = DATAROOT + "output/test_v0/Cont-CC-PermExDc-v0/With_LR_Param_Baseline/Episodic/Param_Sweep/{}/"
    pth_base = DATAROOT + "output/test_v0/NonContexTT/Noncontext_PID_Action_Visits/{}/"
    #sweep_offline(pth_base, "GAC")
    #sweep_offline(pth_base, "SAC")
    #sweep_offline(pth_base, "Reinforce")
    # sweep_offline(pth_base, "ETC")
    sweep_offline(pth_base, "GAC")

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
    SHAREPATH = "output/test_v0/NonContexTT/Noncontext_PID_Action_Visits/"
    pths = {
        "GAC w/ Entropy": [DATAROOT + SHAREPATH + "GAC/param_0/", "limegreen", 3],
        "GAC w/o Entropy": [DATAROOT + SHAREPATH + "GAC/param_28", "C1", 1],
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
    sweep_parameter()
    # compare_algorithms()


