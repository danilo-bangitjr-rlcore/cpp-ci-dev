import matplotlib.pyplot as plt
from utils import *

DATAROOT = "../../out/"


def sweep_offline(pth_base, agent="GAC"):
    print(agent)
    root = pth_base.format(agent)
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    load_exp([axs], param_sweep, root, {}, None)
    plt.legend()
    axs.set_ylim(-50, 2)
    axs.set_ylabel("Performance")
    axs.set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/sweep_{}.png".format(agent), dpi=300, bbox_inches='tight')


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

