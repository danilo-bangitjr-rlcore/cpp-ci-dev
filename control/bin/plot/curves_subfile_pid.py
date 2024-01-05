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


    best_offline(pths, "demo_no_smooth", ylim=[-2, 2])

def stable_gac_test():
    def v0():
        # SHAREPATH = "output/test_v0/NonContexTT/learning_rate/target0/replay0/env_scale_10/"
        # pths = {
        #     # "SAC": [DATAROOT + SHAREPATH + "SAC/param_0/", "C0", 3],
        #     # "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 1],
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 5],
        #     # "GAC-OE": [DATAROOT + SHAREPATH + "GAC-OE/param_2", "C2", 5],
        #     "GAC test": [
        #         DATAROOT + "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay50_batch8/env_scale_10/"
        #         + "GACMH/param_28/", "C2", 5],
        # }
        # best_offline(pths, "test_noncontex", ylim=[-2, 2])
        #
        # SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_10/"
        # pths = {
        #     # "SAC": [DATAROOT + SHAREPATH + "SAC/param_4/", "C0", 1],
        #     # "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_3/", "limegreen", 3],
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 5],
        #     "GAC test": [
        #         DATAROOT + "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay50_batch8/env_scale_10/"
        #         + "GACMH/param_29/", "C2", 5],
        # }
        # best_offline(pths, "test_directAction", ylim=[-2, 2])

        SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10_action_0.01_0.99/"
        pths = {
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_1", "C0", 5],
        }
        best_offline(pths, "test_noncontex_clipaction", ylim=[-2, 2])

        SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10_action_0.01_0.99/"
        pths = {
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_18", "C0", 5],
        }
        best_offline(pths, "test_directAction_clipaction", ylim=[-2, 2])

        # SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch8/env_scale_10/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C0", 5],
        #     "GACPS": [
        #         DATAROOT + "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
        #         + "GACPS/param_46/", "C1", 5],
        #     # "GACIn": [
        #     #     DATAROOT + "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
        #     #     + "GACIn/param_28/", "C2", 3],
        # }
        # best_offline(pths, "test_noncontex", ylim=[-2, 2])
        #
        # SHAREPATH = "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_10/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_21", "C0", 5],
        #     "GACPS": [
        #         DATAROOT + "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
        #         + "GACPS/param_13/", "C1", 5],
        #     # "GACIn": [
        #     #     DATAROOT + "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
        #     #     + "GACIn/param_0/", "C2", 3],
        # }
        # best_offline(pths, "test_directAction", ylim=[-2, 2])
        #
        # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_10", "C0", 5],
        #     "GACPS": [
        #         DATAROOT + "output/test_v0/TTChangeAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
        #         + "GACPS/param_18/", "C1", 5],
        #     # "GACIn": [
        #     #     DATAROOT + "output/test_v0/TTChangeAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
        #     #     + "GACIn/param_37/", "C2", 3],
        # }
        # best_offline(pths, "test_changeActionCont", ylim=[-2, 2])
        #
        # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay5000_batch8/env_scale_1/change_0.01/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_16", "C0", 5],
        #     "GACPS": [
        #         DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/change_0.01/"
        #         + "GACPS/param_17/", "C1", 5],
        #     # "GACIn": [
        #     #     DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/change_0.01/"
        #     #     + "GACIn/param_12/", "C2", 3],
        # }
        # best_offline(pths, "test_changeActionDisc", ylim=[-2, 2])
        #
        # """
        # Reward staying
        # """
        # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteRwdStay/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/change_0.01/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_16", "C0", 5],
        # }
        # best_offline(pths, "test_changeActionRwdStay", ylim=[-2, 2])
        #
        # """
        # Batch normalization
        # """
        # SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_10/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_28", "C0", 5],
        # }
        # best_offline(pths, "test_noncontex_batchNorm", ylim=[-2, 2])
        # SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_10/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_43", "C0", 5],
        # }
        # best_offline(pths, "test_directAction_batchNorm", ylim=[-2, 2])
        # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_37", "C0", 5],
        # }
        # best_offline(pths, "test_changeActionCont_batchNorm", ylim=[-2, 2])
        # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_1/change_0.01/"
        # pths = {
        #     "GAC": [DATAROOT + SHAREPATH + "GAC/param_26", "C0", 5],
        # }
        # best_offline(pths, "test_changeActionDisc_batchNorm", ylim=[-2, 2])
        return

    def v1():
        pths = {
            "GAC": [DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch8/env_scale_10/"
                    + "GAC/param_2", "C0", 5],
            # "GACPS-old": [DATAROOT + "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
            #               + "GACPS/param_46/", "C2", 5],
            "GACPS": [DATAROOT + "output/test_v0/NonContexTT/stable_gac_test/v1/target0/replay5000_batch8/env_scale_10/"
                      + "GACPS/param_82", "C1", 5],
        }
        best_offline(pths, "testv1_noncontex", ylim=[-2, 2])

        pths = {
            "GAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_10/"
                    + "GAC/param_21", "C0", 5],
            # "GACPS-old": [DATAROOT + "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
            #               + "GACPS/param_13/", "C2", 5],
            "GACPS": [DATAROOT + "output/test_v0/TTAction/ConstPID/stable_gac_test/v1/target0/replay5000_batch8/env_scale_10/"
                      + "GACPS/param_10", "C1", 5],
        }
        best_offline(pths, "testv1_directAction", ylim=[-2, 2])

    v0()
    # v1()

def gac_learning_rate():
    def noncontext():
        SHAREPATH = "output/test_v0/NonContexTT/learning_rate_larger_range/target0/replay5000_batch8/env_scale_10/"
        pths = {
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_79/", "C0", 5],
            "GAC-less-lr": [
                DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch8/env_scale_10/"
                + "GAC/param_2/", "C1", 3],
        }
        best_offline(pths, "lr_noncontex", ylim=[-2, 2])

    def direct_action():
        SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate_larger_range/target0/replay5000_batch8/env_scale_10/"
        pths = {
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_10/", "C0", 5],
            "GAC-less-lr": [
                DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay50_batch8/env_scale_10/"
                + "GAC/param_9/", "C1", 3],
        }
        best_offline(pths, "lr_direct_action", ylim=[-2, 2])

    noncontext()
    direct_action()
def gac_parameter_study():
    def noncontext():
        pths = {
            "replay5000-batch32": [DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch32/env_scale_10/"
                    + "GAC/param_46/", "C3", 7],
            "replay5000-batch8": [DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch8/env_scale_10/"
                    + "GAC/param_2/", "C8", 7],
            # "replay100-batch32": [DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay100_batch32/env_scale_10/"
            #         + "GAC/param_46/", "C0", 1],
            # "replay50-batch16": [DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay50_batch16/env_scale_10/"
            #         + "GAC/param_11/", "C1", 3],
            # "replay50-batch8": [DATAROOT + "output/test_v0/NonContexTT/parameter_study/target0/replay50_batch8/env_scale_10/"
            #         + "GAC/param_20/", "C2", 5],
        }
        best_offline(pths, "gac_noncontex", ylim=[-2, 2])

    def direct_action():
        pths = {
            # "replay5000-batch32": [DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay5000_batch32/env_scale_10/"
            #         + "GAC/param_9/", "C3", 7],
            # "replay5000-batch8": [DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_10/"
            #         + "GAC/param_29/", "C8", 7],
            "replay100-batch32": [DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay100_batch32/env_scale_10/"
                    + "GAC/param_21/", "C0", 1],
            "replay50-batch16": [DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay50_batch16/env_scale_10/"
                    + "GAC/param_12/", "C1", 5],
            "replay50-batch8": [DATAROOT + "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay50_batch8/env_scale_10/"
                    + "GAC/param_9/", "C2", 3],
        }
        best_offline(pths, "gac_direct_action", ylim=[-2, 2])

    def change_action():
        pths = {
            "replay5000-batch32": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay5000_batch32/env_scale_1/action_-0.1_0.1/"
                    + "GAC/param_6/", "C3", 7],
            "replay5000-batch8": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
                    + "GAC/param_10/", "C8", 7],
            # "replay100-batch32": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"
            #         + "GAC/param_32/", "C0", 1],
            # "replay50-batch16": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay50_batch16/env_scale_1/action_-0.1_0.1/"
            #         + "GAC/param_41/", "C1", 3],
            # "replay50-batch8": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay50_batch8/env_scale_1/action_-0.1_0.1/"
            #         + "GAC/param_45/", "C2", 5],
        }
        best_offline(pths, "gac_change_action", ylim=[-2, 2])

    def change_action_discrete():
        pths = {
            "replay5000-batch32": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay5000_batch32/env_scale_1/change_0.01/"
                    + "GAC/param_12/", "C3", 7],
            "replay5000-batch8": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay5000_batch8/env_scale_1/change_0.01/"
                    + "GAC/param_16/", "C8", 7],
            # "replay100-batch32": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay100_batch32/env_scale_1/change_0.01/"
            #         + "GAC/param_39/", "C0", 1],
            # "replay50-batch16": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay50_batch16/env_scale_1/change_0.01/"
            #         + "GAC/param_25/", "C1", 3],
            # "replay50-batch8": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay50_batch8/env_scale_1/change_0.01/"
            #         + "GAC/param_43/", "C2", 5],
        }
        best_offline(pths, "gac_change_action_discrete", ylim=[-2, 2])

    # noncontext()
    direct_action()
    # change_action()
    # change_action_discrete()
def constant_pid_target0_replay0():
    def threetank():
        """
        ThreeTanks Direct Action, constant PID
        """
        SHAREPATH = "output/test_v0/ThreeTank/learning_rate/target0/replay0/env_scale_10/"
        pths = {
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_1", "C1", 5],
            "GAC-OE": [DATAROOT + SHAREPATH + "GAC-OE/param_1", "C2", 5],
        }
        best_offline(pths, "best_3tank_replay0_e10", ylim=[-2, 2])
        # reproduce_demo(pths, "best_3tank_replay0_e10", ylim=[-50, 2])

    def noncontext():
        """
        ThreeTanks Direct Action, constant PID
        """
        SHAREPATH = "output/test_v0/NonContexTT/learning_rate/target0/replay0/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_0/", "C0", 3],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 1],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 5],
            # "GAC-OE": [DATAROOT + SHAREPATH + "GAC-OE/param_2", "C2", 5],
        }
        best_offline(pths, "best_noncontex_replay0_e10", ylim=[-2, 2])

    def direct_action():
        """
        ThreeTanks Direct Action, constant PID
        """
        SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay0/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_1/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_3", "C1", 1],
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

    def clip_change_action():
        SHAREPATH = "output/test_v0/TTChangeAction/ClipConstPID/learning_rate/target0/replay0/env_scale_1/action_-0.1_0.1/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_4/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_6/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 1],
        }
        best_offline(pths, "best_changeActionClip_replay0_const_pid_e1_a0.1", ylim=[-50, 2])

        SHAREPATH = "output/test_v0/TTChangeAction/ClipDiscreteConstPID/learning_rate/target0/replay0/env_scale_1/change_0.01/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_3/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_6/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C1", 5],
        }
        best_offline(pths, "best_changeAction_clip_replay0_const_pid_discrete", ylim=[-50, 2])

    def change_action_clip_distribution_param():
        SHAREPATH = "output/test_v0/ThreeTank/clip_distribution_param/target0/replay0/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_0/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_8", "C1", 1],
        }
        best_offline(pths, "best_3tank_clipDistribution_replay0_e10", ylim=[-50, 2])

        SHAREPATH = "output/test_v0/TTAction/ConstPID/clip_distribution_param/target0/replay0/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_0/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_3", "C1", 1],
        }
        best_offline(pths, "best_directAction_clipDistribution_replay0_const_pid_e10", ylim=[-10, 2])

        SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/clip_distribution_param/target0/replay0/env_scale_1/action_-0.1_0.1/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_8/", "C0", 5],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_6/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_5", "C1", 1],
        }
        best_offline(pths, "best_changeAction_clipDistribution_replay0_const_pid_e1_a0.1", ylim=[-10, 2])

    # threetank()
    # noncontext()
    direct_action()
    # change_action_continuous()
    # change_action_discrete()
    # clip_change_action()
    # change_action_clip_distribution_param()

def constant_pid_target0_replay100():
    def threetank():
        """
        ThreeTanks Direct Action, constant PID
        """
        SHAREPATH = "output/test_v0/ThreeTank/learning_rate/target0/replay100_batch32/env_scale_10/"
        pths = {
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C1", 5],
            "GAC-OE": [DATAROOT + SHAREPATH + "GAC-OE/param_0", "C2", 5],
        }
        best_offline(pths, "best_3tank_replay100_batch32_e10", ylim=[-2, 2])

    def noncontext():
        SHAREPATH = "output/test_v0/NonContexTT/learning_rate/target0/replay100_batch32/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_0/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_1", "C1", 5],
            # "GAC-OE": [DATAROOT + SHAREPATH + "GAC-OE/param_1", "C2", 5],
        }
        best_offline(pths, "best_noncontext_replay100_batch32_e10", ylim=[-2, 2])

    def direct_action():
        SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_10/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_4/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_3/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_3", "C1", 5],
        }
        # best_offline(pths, "best_TTAction_replay100_batch32_e10", ylim=[-50, 2])
        best_offline(pths, "best_TTAction_replay100_batch32_e10", ylim=[-2, 2])

    def change_action_contiuous():
        SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"
        pths = {
            "SAC": [DATAROOT + SHAREPATH + "SAC/param_4/", "C0", 1],
            "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
            "GAC": [DATAROOT + SHAREPATH + "GAC/param_0", "C1", 5],
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

    # threetank()
    # noncontext()
    direct_action()
    # change_action_contiuous()
    # change_action_discrete()

def visualize_general():
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
    def change_action_continuous_replay0(target_key):
        file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/visualize/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=None)
        # visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[2950, 3050])
        # visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[3600, 3700])
        # visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[1850, 1950])
        # visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[1000, 1500])
        # visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[500, 1000])
        # visualize_training_info(file, target_key, title="vis_ChangeAction_GAC", threshold=0.99, xlim=[0, 500])

    def change_action_discrete_replay0(target_key):
        target_key.remove("actor_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/visualize/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeAction_discrete_GAC", threshold=0.99, xlim=None, ylim=[-2, 2])
    def direct_action_replay0(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay0/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay0", threshold=0.99, xlim=None, ylim=[-2, 2])

    def direct_action_replay100(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay100/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay100", threshold=0.99, xlim=None, ylim=[-2, 2])

    def clip_distribution_replay0(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/clip_distribution_param/target0/replay0/env_scale_10/GAC/param_3/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay0_clip_distribution", threshold=0.99, xlim=None)


    # change_action_continuous_replay0(copy.deepcopy(target_key))
    # change_action_discrete_replay0(copy.deepcopy(target_key))
    direct_action_replay0(copy.deepcopy(target_key))
    direct_action_replay100(copy.deepcopy(target_key))

    # clip_distribution_replay0(copy.deepcopy(target_key))

def visualize_gac():
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

    def noncontext_replay0(target_key):
        file = DATAROOT + "output/test_v0/NonContexTT/visualize/replay0/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_noncontext_GAC_replay0", threshold=0.99, xlim=None, ylim=[-2, 2])

    def noncontext_replay50(target_key):
        file = DATAROOT + "output/test_v0/NonContexTT/visualize/replay50/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_noncontext_GAC_replay50", threshold=0.99, xlim=None, ylim=[-2, 2])

    def noncontext_replay100(target_key):
        file = DATAROOT + "output/test_v0/NonContexTT/visualize/replay100/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_noncontext_GAC_replay100", threshold=0.99, xlim=None, ylim=[-2, 2])

    def noncontext_replay5000(target_key):
        file = DATAROOT + "output/test_v0/NonContexTT/visualize/replay5000/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_noncontext_GAC_replay5000", threshold=0.99, xlim=None, ylim=[-2, 2])

    def direct_action_replay0(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay0/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay0", threshold=0.99, xlim=None, ylim=[-2, 2])

    def direct_action_replay50(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay50/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay50", threshold=0.99, xlim=None, ylim=[-2, 2])

    def direct_action_replay100(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay100/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay100", threshold=0.99, xlim=None, ylim=[-2, 2])

    def direct_action_replay5000(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay5000/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay5000", threshold=0.99, xlim=None, ylim=[-2, 2])

    def direct_action_replay50_batch16(target_key):
        file = DATAROOT + "output/test_v0/TTAction/ConstPID/visualize/replay50_batch16/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_DirectAction_GAC_replay50_batch16", threshold=0.99, xlim=None, ylim=[-2, 2])

    def change_action_replay0(target_key):
        file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/visualize/replay0/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionCont_GAC_replay0", threshold=0.99, xlim=None, ylim=[-2, 2])

        target_key.remove("actor_info/param2")
        target_key.remove("proposal_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/visualize/replay0/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionDisc_GAC_replay0", threshold=0.99, xlim=None, ylim=[-2, 2])

    def change_action_replay50(target_key):
        file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/visualize/replay50/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionCont_GAC_replay50", threshold=0.99, xlim=None, ylim=[-2, 2])

        target_key.remove("actor_info/param2")
        target_key.remove("proposal_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/visualize/replay50/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionDisc_GAC_replay50", threshold=0.99, xlim=None, ylim=[-2, 2])

    def change_action_replay100(target_key):
        file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/visualize/replay100/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionCont_GAC_replay100", threshold=0.99, xlim=None, ylim=[-2, 2])

        target_key.remove("actor_info/param2")
        target_key.remove("proposal_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/visualize/replay100/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionDisc_GAC_replay100", threshold=0.99, xlim=None, ylim=[-2, 2])

    def change_action_replay5000(target_key):
        file = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/visualize/replay5000/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionCont_GAC_replay5000", threshold=0.99, xlim=None, ylim=[-2, 2])

        target_key.remove("actor_info/param2")
        target_key.remove("proposal_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/visualize/replay5000/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionDisc_GAC_replay5000", threshold=0.99, xlim=None, ylim=[-2, 2])

    def change_action_rwd_stay_replay5000(target_key):
        target_key.remove("actor_info/param2")
        target_key.remove("proposal_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteRwdStay/visualize/replay5000/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionDisc_GAC_replay5000", threshold=0.99, xlim=None, ylim=[-2, 2])

    def change_action_rwd_stay_long(target_key):
        target_key.remove("actor_info/param2")
        target_key.remove("proposal_info/param2")
        file = DATAROOT + "output/test_v0/TTChangeAction/DiscreteRwdStay/visualize/replay50000/GAC/param_0/seed_0"
        visualize_training_info(file, target_key, title="vis_ChangeActionDisc_GAC_long", threshold=0.99, xlim=None, ylim=[-2, 2])

    # noncontext_replay0(copy.deepcopy(target_key))
    # noncontext_replay50(copy.deepcopy(target_key))
    # noncontext_replay100(copy.deepcopy(target_key))
    # noncontext_replay5000(copy.deepcopy(target_key))
    # direct_action_replay0(copy.deepcopy(target_key))
    # direct_action_replay50(copy.deepcopy(target_key))
    # direct_action_replay100(copy.deepcopy(target_key))
    # direct_action_replay5000(copy.deepcopy(target_key))
    # direct_action_replay50_batch16(copy.deepcopy(target_key))
    # change_action_replay0(copy.deepcopy(target_key))
    # change_action_replay50(copy.deepcopy(target_key))
    # change_action_replay100(copy.deepcopy(target_key))
    # change_action_replay5000(copy.deepcopy(target_key))
    # change_action_rwd_stay_replay5000(copy.deepcopy(target_key))
    # change_action_rwd_stay_long(copy.deepcopy(target_key))

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


if __name__ == '__main__':
    # agent_list = ['SAC', 'SimpleAC', 'GAC']
    agent_list = ['GAC']

    # SHAREPATH = "output/test_v0/ThreeTank/learning_rate/target0/replay0/env_scale_10/"
    # SHAREPATH = "output/test_v0/NonContexTT/learning_rate/target0/replay0/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay0/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay0/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/target0/replay0/env_scale_1/change_0.01/"

    # SHAREPATH = "output/test_v0/ThreeTank/learning_rate/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/NonContexTT/learning_rate/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/target0/replay100_batch32/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/target0/replay100_batch32/env_scale_1/change_0.01/"

    SHAREPATH = "output/test_v0/NonContexTT/learning_rate_larger_range/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate_larger_range/target0/replay5000_batch8/env_scale_10/"

    # SHAREPATH = "output/test_v0/TTChangeAction/ClipConstPID/learning_rate/target0/replay0/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ClipDiscreteConstPID/learning_rate/target0/replay0/env_scale_1/change_0.01/"

    # SHAREPATH = "output/test_v0/ThreeTank/clip_distribution_param/target0/replay0/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/clip_distribution_param/target0/replay0/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/clip_distribution_param/target0/replay0/env_scale_1/action_-0.1_0.1/"

    # SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay50_batch16/env_scale_10/"
    # SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay50_batch8/env_scale_10/"

    # SHAREPATH = "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay5000_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay50_batch16/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/parameter_study/target0/replay50_batch8/env_scale_10/"

    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay5000_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay50_batch16/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/parameter_study/target0/replay50_batch8/env_scale_10/"

    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay5000_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay100_batch32/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay50_batch16/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/parameter_study/target0/replay50_batch8/env_scale_10/"

    # agent_list = ['GACMH']
    # SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay50_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay50_batch8/env_scale_10/"

    # agent_list = ['GACPS', 'GACIn']
    # SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/change_0.01/"

    agent_list = ['GAC']
    # SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10_action_0.01_0.99/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0/replay5000_batch8/env_scale_10_action_0.01_0.99/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteRwdStay/stable_gac_test/v0/target0/replay5000_batch8/env_scale_1/change_0.01/"

    # SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_1/action_-0.1_0.1/"
    # SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/stable_gac_test/v0/target0_batchNorm/replay5000_batch8/env_scale_1/change_0.01/"

    # agent_list = ['GACPS']
    # SHAREPATH = "output/test_v0/NonContexTT/stable_gac_test/v1/target0/replay5000_batch8/env_scale_10/"
    # SHAREPATH = "output/test_v0/TTAction/ConstPID/stable_gac_test/v1/target0/replay5000_batch8/env_scale_10/"

    # sweep_parameter(DATAROOT + SHAREPATH, agent_list)

    SHAREPATH = "output/test_v0/NonContexTT/parameter_study/target0/replay50_batch8/env_scale_10/"
    fixed_params_list = {
        "rho": [0.1, 0.2],
        "lr_actor": [0.01, 0.001, 0.0001],
        "lr_critic": [0.01, 0.001, 0.0001],
    }
    sweep_param = "tau" # 0.001
    # draw_sensitivity(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, sweep_param, "sensitivity_tau")
    fixed_params_list = {
        "tau": [0.01, 0.001, 0.0001],
        "lr_actor": [0.01, 0.001, 0.0001],
        "lr_critic": [0.01, 0.001, 0.0001],
    }
    sweep_param = "rho" # 0.1 is better
    # draw_sensitivity(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, sweep_param, "sensitivity_rho")

    fixed_params_list = {
        "lr_actor": [0.01, 0.001, 0.0001],
        "lr_critic": [0.01, 0.001, 0.0001],
    }
    # draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "rho", "tau", "sensitivity_rho_tau") # y: rho, x: tau

    fixed_params_list = {
        "rho": [0.1],
        "tau": [0.001],
    }
    SHAREPATH = "output/test_v0/NonContexTT/learning_rate_larger_range/target0/replay5000_batch8/env_scale_10/"
    # draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "lr_actor", "lr_critic", "sensitivity_lr_noncontex")
    SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate_larger_range/target0/replay5000_batch8/env_scale_10/"
    # draw_sensitivity_2d(DATAROOT + SHAREPATH, 'GAC', fixed_params_list, "lr_actor", "lr_critic", "sensitivity_lr_directaction")


    # demo()
    # stable_gac_test()
    gac_learning_rate()
    # gac_parameter_study()
    # constant_pid_target0_replay0()
    # constant_pid_target0_replay100()

    # visualize_general()
    # visualize_gac()

    # visualize_temp()