from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline
from curves import visualize_training_info


def sweep_parameter():
    pth_base = DATAROOT + "output/test_v0/ReactorEnv/With_LR_Param/GAC_Sweep/{}/"
    sweep_offline(pth_base, "GAC")

def constant_pid_target0_replay0():
    """
    ThreeTanks Direct Action, constant PID
    """
    SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay0/env_scale_10/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_1/", "C0", 5],
        "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 1],
    }
    # best_offline(pths, "best_TTAction_replay0_e10", ylim=[-2, 2])

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


