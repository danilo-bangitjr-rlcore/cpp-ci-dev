import matplotlib.pyplot as plt
from utils import *

DATAROOT = "../../out/"

def learning_rates_offline():
    fix_param = [
        {"lr_constraint": 0.0001},
        {"lr_constraint": 0.00001},
        {"lr_constraint": 0.000001},
        {"lr_constraint": 0.0000001},
    ]
    root = DATAROOT + "output/learning_rate/without_replay/"
    fig, axs = plt.subplots(2, len(fix_param), figsize=(3 * len(fix_param), 4))
    for pi, fp in enumerate(fix_param):
        load_exp(axs[:, pi], sensitivity_curve, root, fp, "lr_critic")
        axs[0, pi].set_title(' '.join('{}={}'.format(key, value) for key, value in fp.items()))
    axs[0, 0].set_ylabel("Performance")
    axs[1, 0].set_ylabel("Constraint")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/learning_rate_sensitivity.png", dpi=300, bbox_inches='tight')


def sweep_offline(pth_base, agent="GAC"):
    print(agent)
    root = pth_base.format(agent)
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    load_exp([axs], param_sweep, root, {}, None)
    plt.legend()
    axs.set_ylabel("Performance")
    axs.set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/param_sweep_{}.png".format(agent), dpi=300, bbox_inches='tight')


def best_offline(pths, title):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    # axins = zoomed_inset_axes(axs, 6, loc=1)
    axs = [axs]
    for label, [pth, c, z] in pths.items():
        setting, returns, constraints = load_param(pth)
        learning_curve(axs[0], returns, label=label, color=c, zorder=z)
    axs[0].set_ylim(-10, 2)
    # axs[0].set_ylim(-1, 1.5)
    axs[0].legend()
    # axs[1].legend()
    axs[0].set_ylabel("Performance")
    # axs[1].set_ylabel("Constraint")
    # axs[1].set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    pth_base = DATAROOT + "output/test_v0/TTChangeAction/ConstPID/learning_rate/without_replay/action_-0.1_0.1/{}/"
    # sweep_offline(pth_base, "SAC")
    # sweep_offline(pth_base, "SimpleAC")
    # sweep_offline(pth_base, "GAC")

    """
    ThreeTanks
    """
    pths = {
        "SAC": [DATAROOT + "output/test_v0/ThreeTank/learning_rate/without_replay/action_0_1/SAC/param_6/", "C0", 3],
        "SimpleAC": [DATAROOT + "output/test_v0/ThreeTank/learning_rate/without_replay/action_0_1/SimpleAC/param_10/", "limegreen", 2],
        "GAC": [DATAROOT + "output/test_v0/ThreeTank/learning_rate/without_replay/action_0_1/GAC/param_8", "C1", 1],
    }
    best_offline(pths, "best_ThreeTanks_no_replay_a01")

    pths = {
        "SAC": [DATAROOT + "output/test_v0/ThreeTank/learning_rate/without_replay/action_0_4/SAC/param_8/", "C0", 3],
        "SimpleAC": [DATAROOT + "output/test_v0/ThreeTank/learning_rate/without_replay/action_0_4/SimpleAC/param_10/", "limegreen", 2],
        "GAC": [DATAROOT + "output/test_v0/ThreeTank/learning_rate/without_replay/action_0_4/GAC/param_8", "C1", 1],
    }
    best_offline(pths, "best_ThreeTanks_no_replay_a04")

    """
    ThreeTanks Direct Action, constant PID
    """
    pths = {
        "SAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/action_0_1/SAC/param_9/", "C0", 2],
        "SimpleAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/action_0_1/SimpleAC/param_11/", "limegreen", 1],
        "GAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/action_0_1/GAC/param_15", "C1", 3],
    }
    best_offline(pths, "best_ThreeTanks_no_replay_const_pid_a01")

    pths = {
        "SAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/action_0_4/SAC/param_1/", "C0", 3],
        "SimpleAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/action_0_4/SimpleAC/param_11/", "limegreen", 2],
        "GAC": [DATAROOT + "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/action_0_4/GAC/param_11", "C1", 1],
    }
    best_offline(pths, "best_ThreeTanks_no_replay_const_pid_a04")

    """
    ThreeTanks Change Action, constant PID
    """
    pths = {
        "SAC": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/learning_rate/without_replay/action_-0.1_0.1/SAC/param_0/", "C0", 3],
        "SimpleAC": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/learning_rate/without_replay/action_-0.1_0.1/SimpleAC/param_3/", "limegreen", 2],
        "GAC": [DATAROOT + "output/test_v0/TTChangeAction/ConstPID/learning_rate/without_replay/action_-0.1_0.1/GAC/param_0", "C1", 1],
    }
    best_offline(pths, "best_changeAction_no_replay_const_pid_c01")

    """
    ThreeTanks Change Action, constant PID, Binning action
    """
    pths = {
        "SAC": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/without_replay/change_0.1/SAC/param_0/", "C0", 3],
        "SimpleAC": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/without_replay/change_0.1/SimpleAC/param_3/", "limegreen", 2],
        "GAC": [DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/without_replay/change_0.1/GAC/param_4", "C1", 1],
    }
    best_offline(pths, "best_changeActionDiscrete_no_replay_const_pid_c01")
