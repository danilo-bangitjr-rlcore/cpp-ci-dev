import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


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

    fig, axs = plt.subplots(2, sharex=True)

    if len(mean.shape) != 2:
        mean = np.expand_dims(mean, axis=0)
        variance = np.expand_dims(variance, axis=0)

    action_dim = mean.shape[0]

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

    fig, axs = plt.subplots(2, sharex=True)

    if len(param1.shape) != 2:
        param1 = np.expand_dims(param1, axis=0)
        param2 = np.expand_dims(param2, axis=0)

    action_dim = param1.shape[0]

    for action_idx in range(action_dim):
        axs[0].plot(param1[action_idx, :], label='dim {}'.format(action_idx))
        axs[1].plot(param2[action_idx, :], label='dim {}'.format(action_idx))

    axs[0].legend(bbox_to_anchor=(1.05, 1.05))

    axs[0].set_ylabel('Param 1')
    axs[1].set_ylabel('Param 2')
    axs[0].set_xlabel('Step')

    remove_spines(axs)
    plt.savefig(save_path / 'action_params.png', bbox_inches='tight')


def make_action_gap_plot(stats, save_path):
    if 'action_gap' not in stats:
        return

    action_gap = stats['action_gap']
    fig, ax = plt.subplots()
    ax.plot(action_gap)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action gap')
    plt.savefig(save_path / 'action_gap.png', bbox_inches='tight')


def make_bellman_error_plot(stats, save_path):
    if 'bellman_error' not in stats:
        return

    bes = stats['bellman_error']

    fig, ax = plt.subplots()
    ax.plot(bes)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('BE')
    plt.savefig(save_path / 'bellman_error.png', bbox_inches='tight')


def make_reward_plot(stats, save_path):
    if 'rewards' not in stats:
        return

    rewards = stats['rewards']

    fig, ax = plt.subplots()
    ax.plot(rewards)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    plt.savefig(save_path / 'rewards.png', bbox_inches='tight')


def make_plots(freezer, stats, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    make_action_mean_variance_plot(freezer, save_path)
    make_param_plot(freezer, save_path)
    make_action_gap_plot(stats, save_path)
    make_bellman_error_plot(stats, save_path)
    make_reward_plot(stats, save_path)

def plot_action_value_alert(plot_info, alert_thresholds, path):
    composite_alerts = plot_info["composite_alert"]
    signal_names = [col_name for col_name in list(plot_info) if col_name not in ["state", "action", "reward", "next_obs", "value", "return", "alert", "alert_trace", "composite_alert"]]
    sensor_names = plot_info["alert"].keys()

    min_range = float('inf')
    # Plot Inidividual Sensor Alerts
    for sensor_name in sensor_names:
        values = plot_info["value"][sensor_name]
        returns = plot_info["return"][sensor_name]
        alerts = plot_info["alert"][sensor_name]
        alert_traces = plot_info["alert_trace"][sensor_name]
        x_indices = list(range(len(values)))
        min_range = min(min_range, len(x_indices))

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

        # Alerts
        for idx in x_indices:
            if alerts[idx]:
                ax[0].axvline(x=idx, color='r', alpha=0.2)
                ax[1].axvline(x=idx, color='r', alpha=0.2)
                ax[2].axvline(x=idx, color='r', alpha=0.2)

        for signal_name in signal_names:
            plot_info[signal_name] = plot_info[signal_name][-len(returns):]
            ax[0].plot(x_indices, plot_info[signal_name], label=signal_name, alpha=1.0)
        setpoint = [350 for i in x_indices]
        ax[0].plot(x_indices, setpoint, c="k", label="Setpoint", alpha=1.0)
        ax[0].set_title("Signals", pad=0)

        ax[1].plot(x_indices, values, label="{} Q(s,a)".format(sensor_name), c="b", alpha=1.0)
        ax[1].plot(x_indices, returns, label="Observed Return", c="g", alpha=1.0)
        ax[1].set_title("{} Q(s,a) vs. Observed Partial Returns".format(sensor_name), pad=0)

        alert_thresh = [alert_thresholds[sensor_name] for i in x_indices]
        ax[2].plot(x_indices, alert_thresh, c="k", label="Alert Threshold", alpha=1.0)
        ax[2].plot(x_indices, alert_traces, label="{} Trace".format(sensor_name), c="m", alpha=1.0)
        ax[2].set_title("{} Q(s,a) vs. Observed Partial Returns Percentage Difference Trace".format(sensor_name), pad=0)

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.xlabel("Time Step")
        fig.savefig(path / "{}_Alerts_Summary_Plot.png".format(sensor_name))
        plt.close()

    # Plot Composite Alert
    x_indices = list(range(min_range))
    composite_alerts = composite_alerts[-min_range:]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 12))

    # Alerts
    for idx in x_indices:
        if composite_alerts[idx]:
            ax[0].axvline(x=idx, color='r', alpha=0.2)
            ax[1].axvline(x=idx, color='r', alpha=0.2)

    for signal_name in signal_names:
        plot_info[signal_name] = plot_info[signal_name][-min_range:]
        ax[0].plot(x_indices, plot_info[signal_name], label=signal_name, alpha=1.0)
    setpoint = [350 for i in x_indices]
    ax[0].plot(x_indices, setpoint, c="k", label="Setpoint", alpha=1.0)
    ax[0].set_title("Signals", pad=0)

    colors = ["purple", "green", "grey", "dodgerblue", "lawngreen", "fuchsia", "olive", "teal"]
    counter = 0
    for sensor_name in sensor_names:
        alert_thresh = [alert_thresholds[sensor_name] for i in x_indices]
        ax[1].plot(x_indices, alert_thresh, c=colors[counter], linestyle="dashed", label="{} Alert Threshold".format(sensor_name), alpha=1.0)
        ax[1].plot(x_indices, plot_info["alert_trace"][sensor_name][-min_range:], c=colors[counter], linestyle="solid", label="{} Trace".format(sensor_name), alpha=1.0)
        ax[1].set_title("{} Q(s,a) vs. Observed Partial Returns Percentage Difference Trace".format(sensor_name), pad=0)
        counter += 1

    ax[0].legend()
    ax[1].legend()
    plt.xlabel("Time Step")
    fig.savefig(path / "Composite_Alert_Summary_Plot.png")
    plt.close()

def visualize_actor_critic(states, actions, q_values, actor_params, env, path, prefix, epoch):
    # Currently assuming 'ReseauAnytime' state constructor
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    mins = [0, action_space_low[0], obs_space_low[0], obs_space_low[1], action_space_low[0], action_space_low[0], action_space_low[0], action_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[1], obs_space_low[1], obs_space_low[1], obs_space_low[1], 0, 0]
    maxs = [1, action_space_high[0], obs_space_high[0], obs_space_high[1], action_space_high[0], action_space_high[0], action_space_high[0], action_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[1], obs_space_high[1], obs_space_high[1], obs_space_high[1], 1, 1]

    for i in range(len(states)):
        curr_state = states[i]
        curr_alpha = actor_params[i][0]
        curr_beta = actor_params[i][1]

        # Create graph title
        for j in [8, 9, 10, 11, 12, 13, 14, 15]:
            curr_state[j] = curr_state[j] * (maxs[j] - mins[j])
        
        for j in [1, 2, 3, 4, 5, 6, 7]:
            curr_state[j] = curr_state[j] * (maxs[j] - mins[j]) + mins[j]
        
        ax_title = 'ORP: '
        for j in [2, 8, 9, 10, 11]:
            ax_title += "{:.3e} ".format(curr_state[j])

        ax_title += '\nFPM: '
        for j in [1, 4, 5, 6, 7]:
            ax_title += "{:.3e} ".format(curr_state[j])

        ax_title += '\nFlow Rate: '
        for j in [3, 12, 13, 14, 15]:
            ax_title += "{:.3e} ".format(curr_state[j])

        ax_title += '\nCountdown: {}'.format(curr_state[16])
        ax_title += '\nDecision Step: {}'.format(curr_state[17])

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 12))
        ax[0].plot(
                actions,
                beta.pdf(actions, curr_alpha, curr_beta),
                label="Actor Policy",
                c="r",
            )
        ax[0].set_title("Actor Policy", pad=0)
        ax[1].plot(actions, q_values[i], label="Q Function", c="b", alpha=1.0)
        ax[1].set_title("Q-Function", pad=20)
        plt.xlabel("Action Space")
        fig.suptitle(ax_title)
        # ax.legend()
        fig.savefig(path / "{}_epoch_{}_test_state_{}_Summary_Plots.png".format(prefix, epoch, i))
        plt.close()


