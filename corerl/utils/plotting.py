import numpy as np
import matplotlib.pyplot as plt


def remove_spines(axs, spines=["top", "right"], has_subplots=True):
    def _remove_spines(ax, spines):
        for spine in spines:
            if spine in ax.spines:
                ax.spines[spine].set_visible(False)

    if has_subplots:
        for ax in axs.flat:
            _remove_spines(ax, spines)
    else:
        _remove_spines(axs, spines)

def make_action_mean_variance_plot(freezer, save_path):
    action_info = freezer['action_info']

    mean = []
    variance = []
    for info in action_info:
        mean.append(info['mean'])
        variance.append(info['variance'])

    # transpose to make the actions the action dimensions the rows, and the iterations the columns
    mean = np.array(mean).squeeze().T
    variance = np.array(variance).squeeze().T

    assert mean.shape[0] == variance.shape[0]
    action_dim = mean.shape[0]

    fig, axs = plt.subplots(2, sharex=True)
    for action_idx in range(action_dim):
        axs[0].plot(mean[action_idx, :], label='dim {}'.format(action_idx))
        axs[1].plot(variance[action_idx, :], label='dim {}'.format(action_idx))

    axs[0].legend(bbox_to_anchor=(1.05, 1.05))

    axs[0].set_ylabel('Mean')
    axs[1].set_ylabel('Variance')
    axs[0].set_xlabel('Step')

    remove_spines(axs)
    plt.savefig(save_path / 'action_mean_variance.png', bbox_inches='tight')


def make_param_plot(freezer, save_path):
    action_info = freezer['action_info']

    param1 = []
    param2 = []
    for info in action_info:
        param1.append(info['param1'])
        param2.append(info['param2'])

    # transpose to make the actions the action dimensions the rows, and the iterations the columns
    param1 = np.array(param1).squeeze().T
    param2 = np.array(param2).squeeze().T

    assert param1.shape[0] == param2.shape[0]
    action_dim = param1.shape[0]

    fig, axs = plt.subplots(2, sharex=True)
    for action_idx in range(action_dim):
        axs[0].plot(param1[action_idx, :], label='dim {}'.format(action_idx))
        axs[1].plot(param2[action_idx, :], label='dim {}'.format(action_idx))

    axs[0].legend(bbox_to_anchor=(1.05, 1.05))

    axs[0].set_ylabel('Param 1')
    axs[1].set_ylabel('Param 2')
    axs[0].set_xlabel('Step')

    remove_spines(axs)
    plt.savefig(save_path / 'action_params.png', bbox_inches='tight')

def make_action_gap_plot(freezer, save_path):
    action_gap = freezer['action_gap']

    fig, ax = plt.subplots()
    ax.plot(action_gap)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action gap')
    plt.savefig(save_path / 'action_gap.png', bbox_inches='tight')

def make_plots(freezer, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    make_action_mean_variance_plot(freezer, save_path)
    make_param_plot(freezer, save_path)
    make_action_gap_plot(freezer, save_path)




