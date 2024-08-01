import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from corerl.agent.utils import get_test_state_qs_and_policy_params

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
    plt.close()


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
    plt.close()


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
    plt.close()


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
    plt.close()


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
    plt.close()

def make_alerts_plots(stats, path):
    if 'raw_actions' not in stats:
        return

    actions_taken = stats["raw_actions"]
    endo_obs = stats["raw_endo_obs"]

    composite_alerts = stats["composite_alerts"]
    individual_alerts = stats["individual_alerts"]
    alert_traces = stats["alert_traces"]
    values = stats["alert_values"]
    returns = stats["alert_returns"]
    trace_thresholds = stats["alert_trace_thresholds"]

    min_steps = float('inf')
    min_steps = min(min_steps, len(composite_alerts))
    for action_name in actions_taken:
        min_steps = min(min_steps, len(actions_taken[action_name]))
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            min_steps = min(min_steps, len(individual_alerts[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(alert_traces[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(values[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(returns[alert_type][cumulant_name]))
    time_steps = list(range(min_steps))

    # Plot Inidividual Sensor Alerts
    # Top subplot: Actions and Endogenous Variables over time
    # Middle subplot: GVF for a given endogenous variable vs. Observed partial returns for the given endogenous variable
    # Bottom subplot: Trace of absolute difference between GVF and partial return
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

            # Individual Alerts - Plot vertical red lines in each subplot at time steps where there are alerts for the given cumulant
            for step in time_steps:
                if individual_alerts[alert_type][cumulant_name][step]:
                    ax[0].axvline(x=step, color='r', alpha=0.2)
                    ax[1].axvline(x=step, color='r', alpha=0.2)
                    ax[2].axvline(x=step, color='r', alpha=0.2)

            # Plot actions and endogenous variables
            for action_name in actions_taken:
                action_type_actions = actions_taken[action_name][:min_steps]
                ax[0].plot(time_steps, action_type_actions, label=action_name, alpha=1.0)
            for endo_obs_name in endo_obs:
                obs_type_observations = endo_obs[endo_obs_name][:min_steps]
                ax[0].plot(time_steps, obs_type_observations, label=endo_obs_name, alpha=1.0)
            ax[0].set_title("Actions and Endogenous Variables", pad=0)

            # Plot GVF and Partial Return of cumulant_name
            cumulant_name_values = values[alert_type][cumulant_name][:min_steps]
            cumulant_name_returns = returns[alert_type][cumulant_name][:min_steps]
            ax[1].plot(time_steps, cumulant_name_values, label="{} Q(s,a)".format(cumulant_name), c="b", alpha=1.0)
            ax[1].plot(time_steps, cumulant_name_returns, label="Partial Return", c="g", alpha=1.0)
            ax[1].set_title("{} Q(s,a) vs. Observed Partial Returns".format(cumulant_name), pad=0)

            # Plot alert trace
            cumulant_name_traces = alert_traces[alert_type][cumulant_name][:min_steps]
            thresh_list = [trace_thresholds[alert_type][cumulant_name] for _ in range(min_steps)]
            ax[2].plot(time_steps, thresh_list, c="k", label="Trace Threshold", alpha=1.0)
            ax[2].plot(time_steps, cumulant_name_traces, label="{} Trace".format(cumulant_name), c="m", alpha=1.0)
            ax[2].set_title("{} Q(s,a) vs. Observed Partial Returns Absolute Difference Trace".format(cumulant_name), pad=0)

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            plt.xlabel("Time Step")
            fig.savefig(path / "{}_Alerts_Summary_Plot.png".format(cumulant_name))
            plt.close()

    # Plot Composite Alert
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 12))

    # Alerts
    for step in time_steps:
        if composite_alerts[step]:
            ax[0].axvline(x=step, color='r', alpha=0.2)
            ax[1].axvline(x=step, color='r', alpha=0.2)

    # Plot actions and endogenous variables
    for action_name in actions_taken:
        action_type_actions = actions_taken[action_name][:min_steps]
        ax[0].plot(time_steps, action_type_actions, label=action_name, alpha=1.0)
    for endo_obs_name in endo_obs:
        obs_type_observations = endo_obs[endo_obs_name][:min_steps]
        ax[0].plot(time_steps, obs_type_observations, label=endo_obs_name, alpha=1.0)
    ax[0].set_title("Actions and Endogenous Variables", pad=0)

    # Plot all traces and thresholds
    colors = ["purple", "green", "grey", "dodgerblue", "lawngreen", "fuchsia", "olive", "teal"]
    counter = 0
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            thresh_list = [trace_thresholds[alert_type][cumulant_name] for _ in range(min_steps)]
            cumulant_name_traces = alert_traces[alert_type][cumulant_name][:min_steps]
            ax[1].plot(time_steps, thresh_list, c=colors[counter], linestyle="dashed", label="{} Threshold".format(cumulant_name), alpha=1.0)
            ax[1].plot(time_steps, cumulant_name_traces, c=colors[counter], linestyle="solid", label="{} Trace".format(cumulant_name), alpha=1.0)
            counter += 1
    ax[1].set_title("All Alert Traces and Thresholds", pad=0)

    ax[0].legend()
    ax[1].legend()
    plt.xlabel("Time Step")
    fig.savefig(path / "Composite_Alert_Summary_Plot.png")
    plt.close()

def make_reseau_actor_critic_plot(states, actions, q_values, ensemble_q_values, actor_params, env, path, prefix, epoch):
    # Currently assuming 'ReseauAnytime' state constructor
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    mins = [0, action_space_low[0], obs_space_low[0], obs_space_low[1], action_space_low[0], action_space_low[0], action_space_low[0], action_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[1], obs_space_low[1], obs_space_low[1], obs_space_low[1], 0, 0]
    maxs = [1, action_space_high[0], obs_space_high[0], obs_space_high[1], action_space_high[0], action_space_high[0], action_space_high[0], action_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[1], obs_space_high[1], obs_space_high[1], obs_space_high[1], 1, 1]
    ensemble = len(ensemble_q_values)
    for i in range(0, len(states)):
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

        # ax_title += '\nCountdown: {}'.format(curr_state[16])
        # ax_title += '\nDecision Step: {}'.format(curr_state[17])

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 12))

        # Plot Actor Policy
        ax[0].plot(
                actions,
                beta.pdf(actions, curr_alpha, curr_beta),
                label="Actor Policy",
                c="r",
            )
        ax[0].set_title("Actor Policy", pad=0)

        # Plot each critic in the ensemble
        ax[1].plot(actions, q_values[i], label="Q Function", c="b", alpha=1.0)
        for j in range(ensemble):
            ax[1].plot(actions, ensemble_q_values[j][i], alpha=0.2)

        ax[1].set_title("Q-Function", pad=20)
        plt.xlabel("Action Space")
        fig.suptitle(ax_title)
        # ax.legend()
        fig.savefig(path / "{}_epoch_{}_test_state_{}_Summary_Plots.png".format(prefix, epoch, i))
        plt.close()

def make_plots(freezer, stats, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    #make_alerts_plots(stats, save_path)
    #make_action_mean_variance_plot(freezer, save_path)
    #make_param_plot(freezer, save_path)
    #make_action_gap_plot(stats, save_path)
    #make_bellman_error_plot(stats, save_path)
    #make_reward_plot(stats, save_path)

def make_actor_critic_plots(agent, env, plot_transitions, prefix, iteration, save_path):
    test_states, test_actions, q_values, ensemble_q_values, actor_params = get_test_state_qs_and_policy_params(agent, plot_transitions)
    make_reseau_actor_critic_plot(test_states, test_actions, q_values, ensemble_q_values, actor_params, env, save_path, prefix, iteration)

