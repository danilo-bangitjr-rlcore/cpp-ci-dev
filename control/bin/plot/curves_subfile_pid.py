from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline
from curves import visualize_training_info


def sweep_parameter(pth_base):
    sweep_offline(pth_base+"/SAC/", "SAC")
    sweep_offline(pth_base+"/SimpleAC/", "SimpleAC")
    sweep_offline(pth_base+"/GAC/", "GAC")

def demo():
    """
    ThreeTanks
    """
    pths = {
        "Baseline":[DATAROOT + "baseline/reproduce_new_reward_no_smooth/param_0/", "C0", 2],
    }
    reproduce_demo(pths, "reproduce", ylim=[-1200, 2])
    pths = {
        "Baseline":[DATAROOT + "baseline/reproduce_new_reward_no_smooth/param_0/", "C0", 2],
        "New":[DATAROOT + "output/test_v0/ThreeTank/demo/target0/replay0/env_scale_10/GAC/param_1/", "C1", 3]
    }
    reproduce_demo(pths, "demo_compare", ylim=[-1200, 2])
    reproduce_demo(pths, "demo_compare_zoomin", ylim=[-50, 2])

def constant_pid_target0_replay0():
    def direct_action():
        """
        ThreeTanks Direct Action, constant PID
        """
        SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay0/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_1/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 1],
        }
        best_offline(pths, "best_TTAction_replay0_e10", ylim=[-2, 2])

    def change_action_continuous():
        """
        ThreeTanks Change Action, constant PID
        """
        SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay0/env_scale_1/action_-0.1_0.1/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_6/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_6/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_1", "C1", 1],
        }
        best_offline(pths, "best_changeAction_replay0_const_pid_e1_a0.1", ylim=[-50, 2])

    def change_action_discrete():
        SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/target0/replay0/env_scale_1/change_0.01/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_3/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_3/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_3", "C1", 5],
        }
        best_offline(pths, "best_changeAction_replay0_const_pid_discrete", ylim=[-50, 2])

    direct_action()
    change_action_continuous()
    change_action_discrete()

def constant_pid_target0_replay100():
    def direct_action():
        SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_4/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_3/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 5],
        }
        best_offline(pths, "best_TTAction_replay100_batch32_e10", ylim=[-50, 2])

    def change_action_contiuous():
        SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_4/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 5],
        }
        best_offline(pths, "best_changeAction_replay100_batch32_const_pid_e1_a0.1", ylim=[-50, 2])

    def change_action_discrete():
        SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/target0/replay100_batch32/env_scale_1/change_0.01/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_3/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 6],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_7", "C1", 5],
            # "SAC": [DATAROOT + SHAREPATH + "SAC/param_0/", "C0", 1],
            # "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 6],
            # "GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C1", 5],
        }
        best_offline(pths, "best_changeAction_replay100_batch32_const_pid_e1_discrete", ylim=[-50, 2])

    direct_action()
    change_action_contiuous()
    change_action_discrete()
def visualize():
    target_key = [
        "actor_info/param1",
        "actor_info/param2",
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
    file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/visualize/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=None)
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[2950, 3050])
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[3600, 3700])
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[1850, 1950])
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[1000, 1500])
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[500, 1000])
    visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[0, 500])

    file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/visualize/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_ChangeAction_discrete_GAC", threshold=0.99, xlim=None)

    file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay0/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay0", threshold=0.99, xlim=None)

    file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay100/GAC/param_0/seed_0"
    visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay100", threshold=0.99, xlim=None)


if __name__ == '__main__':
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay0/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay0/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/target0/replay0/env_scale_1/change_0.01/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/target0/replay100_batch32/env_scale_1/change_0.01/"
    # pth_base = DATAROOT + SHAREPATH
    # sweep_parameter(pth_base)

    # demo()
    constant_pid_target0_replay0()
    constant_pid_target0_replay100()
    # visualize()