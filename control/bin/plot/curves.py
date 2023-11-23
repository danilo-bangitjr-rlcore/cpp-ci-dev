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


def best_offline(pths, title, ylim):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    # axins = zoomed_inset_axes(axs, 6, loc=1)
    axs = [axs]
    for label, [pth, c, z] in pths.items():
        setting, returns, constraints = load_param(pth)
        learning_curve(axs[0], returns, label=label, color=c, zorder=z)
    axs[0].set_ylim(ylim)
    axs[0].legend()
    # axs[1].legend()
    axs[0].set_ylabel("Performance")
    # axs[1].set_ylabel("Constraint")
    # axs[1].set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')

def best_offline_per_run(pth, title):
    setting, returns, constraints = load_param(pth)
    fig, axs = plt.subplots(1, len(returns), figsize=(3*len(returns), 3))
    if len(returns) == 1:
        axs = [axs]
    learning_curve_per_run(axs, returns)
    # axs[0].set_ylim(-10, 2)
    axs[0].legend()
    axs[0].set_ylabel("Performance")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')


def reproduce_demo(pths, title, ylim):
    def recover_paper_data(returns):
        # denormalization
        returns = returns * 8 - 8
        # smooth
        smoothed = np.zeros(returns.shape)
        smoothed[:, :5] = returns[:, :5]
        for step in range(5, returns.shape[1]):
            smoothed[:, step] = returns[:, step]
            smoothed[:, step] = smoothed[:, step - 4: step + 1].mean()
        return smoothed
    # fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    axs = [axs]
    for label, [pth, c, z] in pths.items():
        setting, returns, constraints = load_param(pth)#, pick_seed=["seed_0", "seed_42"])
        returns = recover_paper_data(returns)
        learning_curve(axs[0], returns, label=label, color=c, zorder=z)
        # constraints = recover_paper_data(constraints)
        # learning_curve(axs[1], constraints, label=label, color=c, zorder=z)
    axs[0].set_ylim(ylim)
    axs[0].legend()
    # axs[1].legend()
    axs[0].set_ylabel("Performance")
    # axs[1].set_ylabel("Constraint")
    # axs[1].set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    pth_base = DATAROOT + "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/without_replay/env_scale_1/change_0.01/{}/"
    # sweep_offline(pth_base, "SAC")
    # sweep_offline(pth_base, "SimpleAC")
    # sweep_offline(pth_base, "GAC")

    """
    ThreeTanks
    """
    pths = {
        "Baseline":[DATAROOT + "baseline/reproduce_new_reward_no_smooth/param_0/", "C0", 2],
    }
    reproduce_demo(pths, "reproduce", ylim=[-1200, 2])
    pths = {
        "Baseline":[DATAROOT + "baseline/reproduce_new_reward_no_smooth/param_0/", "C0", 2],
        "New":[DATAROOT + "output/test_v0/ThreeTank/demo/without_replay/env_scale_10/GAC/param_1/", "C1", 3]
    }
    reproduce_demo(pths, "demo_compare", ylim=[-1200, 2])
    reproduce_demo(pths, "demo_compare_zoomin", ylim=[-50, 2])

    """
    ThreeTanks Direct Action, constant PID
    """
    SHAREPATH = "output/test_v0/TTAction/ConstPID/learning_rate/without_replay/env_scale_10/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_1/", "C0", 5],
        "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_0/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_2", "C1", 1],
    }
    # best_offline(pths, "best_TTAction_replay0_e10", ylim=[-2, 2])

    """
    ThreeTanks Change Action, constant PID
    """
    SHAREPATH = "output/test_v0/TTChangeAction/ConstPID/learning_rate/without_replay/env_scale_1/action_-0.1_0.1/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_6/", "C0", 5],
        "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_6/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_1", "C1", 1],
    }
    # best_offline(pths, "best_changeAction_replay0_const_pid_e1_a0.1", ylim=[-50, 2])

    SHAREPATH = "output/test_v0/TTChangeAction/DiscreteConstPID/learning_rate/without_replay/env_scale_1/change_0.01/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_3/", "C0", 5],
        "SimpleAC": [DATAROOT + SHAREPATH + "SimpleAC/param_3/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_3", "C1", 1],
    }
    # best_offline(pths, "best_changeAction_replay0_const_pid_e1", ylim=[-10, 2])
