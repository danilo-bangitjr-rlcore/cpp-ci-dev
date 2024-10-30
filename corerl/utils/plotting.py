import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
from matplotlib.path import Path
from corerl.agent.utils import get_test_state_qs_and_policy_params

import corerl.utils.nullable as nullable

def remove_spines(
    axs,
    spines: list[str] | None = None,
    has_subplots=True,
):
    spines = nullable.default(spines, lambda: ["top", "right"])
    def _remove_spines(ax):
        for spine in spines:
            if spine in ax.spines:
                ax.spines[spine].set_visible(False)

    if has_subplots:
        for ax in axs.flat:
            _remove_spines(ax)
    else:
        _remove_spines(axs)


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
        param1.append(info['param1'].cpu())
        param2.append(info['param2'].cpu())

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
    axs[0].set_yscale('log')
    axs[1].set_ylabel('Param 2')
    axs[1].set_yscale('log')
    axs[0].set_xlabel('Step')

    remove_spines(axs)
    plt.savefig(save_path / 'action_params.png', bbox_inches='tight')
    plt.close()


def make_action_gap_plot(stats, save_path, prefix):
    if 'action_gap' not in stats:
        return

    action_gap = stats['action_gap']
    fig, ax = plt.subplots()
    ax.plot(action_gap)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action gap')
    plt.savefig(save_path / '{}_action_gap.png'.format(prefix), bbox_inches='tight')
    plt.close()


def make_q_estimation_plot(stats, save_path, prefix):
    if 'q_estimation_max' not in stats:
        return

    qs_max = stats['q_estimation_max']
    qs_min = stats['q_estimation_min']
    qs_med = stats['q_estimation_median']
    qs_avg = stats['q_estimation_avg']
    fig, axs = plt.subplots(4, sharex=True)
    axs[0].plot(qs_max)
    axs[0].set_title("Q Estimation - Max")
    axs[1].plot(qs_min)
    axs[1].set_title("Q Estimation - Min")
    axs[2].plot(qs_med)
    axs[2].set_title("Q Estimation - Median")
    axs[3].plot(qs_avg)
    axs[3].set_title("Q Estimation - Average")

    fig.supxlabel('Step')
    fig.supylabel('Q')
    plt.tight_layout()
    plt.savefig(save_path / '{}_q_estimation.png'.format(prefix), bbox_inches='tight')
    plt.close()

def make_policy_improvement_plot(stats, save_path, prefix):
    if 'policy_improvements' not in stats:
        return

    pi_improve = stats['policy_improvements']
    greedy_gap = stats['greedy_gaps']
    fig, ax = plt.subplots(1)
    ax.axhline(y=0., ls='--', c='grey')
    ax.plot(pi_improve, c='C0', label='Policy improvement')
    ax.plot(greedy_gap, c='C1', label='Greedification gap')

    fig.supxlabel('Step')
    fig.supylabel(r'$\Delta Q$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / '{}_policy_improvement.png'.format(prefix), bbox_inches='tight')
    plt.close()

def make_counterfactual_plot(stats, save_path, prefix):
    if 'efficiency_predictions' not in stats:
        return

    efficiency_predictions = stats['efficiency_predictions']
    fig, ax = plt.subplots(1)
    # ax.axhline(y=0., ls='--', c='grey')
    ax.plot(efficiency_predictions, c='C0', label='Predicted efficiency')

    fig.supxlabel('Step')
    fig.supylabel('Averaged Efficiency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / '{}_counterfactual.png'.format(prefix), bbox_inches='tight')
    plt.close()

def make_bellman_error_plot(stats, save_path, prefix):
    if 'bellman_error' not in stats:
        return

    bes = stats['bellman_error']
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 12))
    ax[0].axhline(y=0., ls='--', c='grey')
    ax[0].plot(bes)
    remove_spines(ax[0], has_subplots=False)
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('BE')

    bes_delta = stats['ibe_change']
    ax[1].axhline(y=0., ls='--', c='grey')
    ax[1].plot(bes_delta)
    remove_spines(ax[1], has_subplots=False)
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel(r'$\delta$BE')

    bes_delta_smoothed = stats['ibe_smoothed_change']
    ax[2].axhline(y=0., ls='--', c='grey')
    ax[2].plot(bes_delta_smoothed)
    remove_spines(ax[2], has_subplots=False)
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel(r'Smoothed $\delta$BE')

    plt.savefig(save_path / '{}_bellman_error.png'.format(prefix), bbox_inches='tight')
    plt.close()

def make_td_error_plot(stats, save_path, prefix):
    if 'td_error' not in stats:
        return

    tdes = stats['td_error']
    fig, ax = plt.subplots()
    ax.axhline(y=0., ls='--', c='grey')
    ax.plot(tdes)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('TDE')
    # ax.set_yscale('log')
    plt.savefig(save_path / '{}_td_error.png'.format(prefix), bbox_inches='tight')
    plt.close()

def make_cumulative_reward_plot(stats, save_path):
    if 'reward_sums' not in stats:
        return

    reward_sums = stats['reward_sums']
    avg_reward = stats['avg_reward']

    fig, ax = plt.subplots()
    ax.plot(reward_sums)
    remove_spines(ax, has_subplots=False)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_yscale('symlog')
    ax.set_title("Avg Reward: {}".format(avg_reward))
    plt.savefig(save_path / 'cumulative_reward.png', bbox_inches='tight')
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

def make_train_test_loss_plot(stats, save_path):
    train_losses = stats["train_losses"]
    test_losses = stats["test_losses"]
    _, ax = plt.subplots(figsize=(12, 12))
    ax.plot(train_losses, linestyle="solid", label="train loss", alpha=1.0)
    ax.plot(test_losses, linestyle="dotted", label="test loss", alpha=1.0)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(save_path / 'critic_offline_train_test_losses.png', bbox_inches='tight')
    plt.close()

def make_trace_alerts_plots(stats, path):
    actions_taken = stats["raw_actions"]
    endo_obs = stats["raw_endo_obs"]

    composite_alerts = stats["composite_alerts"]
    individual_alerts = stats["individual_alerts"]
    alert_traces = stats["alert_traces"]
    values = stats["alert_values"]
    returns = stats["alert_returns"]
    trace_thresholds = stats["alert_trace_thresholds"]

    min_steps = len(composite_alerts)
    for action_name in actions_taken:
        min_steps = min(min_steps, len(actions_taken[action_name]))
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            min_steps = min(min_steps, len(individual_alerts[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(alert_traces[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(values[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(returns[alert_type][cumulant_name]))
    time_steps = list(range(min_steps))

    # Plot Individual Sensor Alerts
    # Top subplot: Actions and Endogenous Variables over time
    # Middle subplot: GVF for a given endogenous variable vs. Observed partial returns for the given endogenous variable
    # Bottom subplot: Trace of absolute difference between GVF and partial return
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

            # Individual Alerts -
            # Plot vertical red lines in each subplot at time steps where there are alerts for the given cumulant
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
            ax[2].set_title(f"{cumulant_name} Q(s,a) vs. Observed Partial Returns Absolute Difference Trace", pad=0)

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            plt.xlabel("Time Step")
            fig.savefig(path / "{}_Trace_Alerts_Summary_Plot.png".format(cumulant_name))
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
            thresh_label = f'{cumulant_name} Threshold'
            ax[1].plot(time_steps, thresh_list, c=colors[counter], linestyle="dashed", label=thresh_label, alpha=1.0)

            trace_label = f'{cumulant_name} Trace'
            ax[1].plot(time_steps, cumulant_name_traces, c=colors[counter], linestyle="solid", label=trace_label, alpha=1.0) # noqa: E501
            counter += 1

    ax[1].set_title("All Alert Traces and Thresholds", pad=0)
    ax[0].legend()
    ax[1].legend()
    plt.xlabel("Time Step")
    fig.savefig(path / "Composite_Alert_Summary_Plot.png")
    plt.close()

def make_uncertainty_alerts_plots(stats, path):
    actions_taken = stats["raw_actions"]
    endo_obs = stats["raw_endo_obs"]

    composite_alerts = stats["composite_alerts"]
    individual_alerts = stats["individual_alerts"]
    alert_traces = stats["alert_traces"]
    alert_trace_thresholds = stats["alert_trace_thresholds"]
    values = stats["alert_values"]
    returns = stats["alert_returns"]
    std_traces = stats["std_traces"]
    std_trace_thresholds = stats["std_trace_thresholds"]

    min_steps = len(composite_alerts)
    for action_name in actions_taken:
        min_steps = min(min_steps, len(actions_taken[action_name]))
    for endo_obs_name in endo_obs:
        min_steps = min(min_steps, len(endo_obs[endo_obs_name]))
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            min_steps = min(min_steps, len(individual_alerts[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(alert_traces[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(values[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(returns[alert_type][cumulant_name]))
            min_steps = min(min_steps, len(std_traces[alert_type][cumulant_name]))
    time_steps = list(range(min_steps))

    # Plot Individual Sensor Alerts
    # Top subplot: Actions and Endogenous Variables over time
    # Second subplot: GVF evaluated at state-action pairs encountered online vs. Observed Partial Returns
    # Third subplot: GVF Ensemble STD
    # Bottom subplot: Alert trace
    for alert_type in individual_alerts:
        for cumulant_name in individual_alerts[alert_type]:
            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

            # Individual Alerts -
            # Plot vertical red lines in each subplot at time steps where there are alerts for the given cumulant
            for step in time_steps:
                if individual_alerts[alert_type][cumulant_name][step]:
                    ax[0].axvline(x=step, color='r', alpha=0.2)
                    ax[1].axvline(x=step, color='r', alpha=0.2)
                    ax[2].axvline(x=step, color='r', alpha=0.2)
                    ax[3].axvline(x=step, color='r', alpha=0.2)

            # Plot actions and endogenous variables
            for action_name in actions_taken:
                action_type_actions = actions_taken[action_name][:min_steps]
                ax[0].plot(time_steps, action_type_actions, label=action_name, alpha=1.0)
            for endo_obs_name in endo_obs:
                obs_type_observations = endo_obs[endo_obs_name][:min_steps]
                ax[0].plot(time_steps, obs_type_observations, label=endo_obs_name, alpha=1.0)
            ax[0].set_title("Actions and Endogenous Variables", pad=0)

            # GVF evaluated at state-action pairs encountered online vs. Observed Partial Returns
            cumulant_name_values = values[alert_type][cumulant_name][:min_steps]
            cumulant_name_returns = returns[alert_type][cumulant_name][:min_steps]
            ax[1].plot(time_steps, cumulant_name_values, label="{} Q(s,a)".format(cumulant_name), c="b", alpha=1.0)
            ax[1].plot(time_steps, cumulant_name_returns, label="Partial Return", c="g", alpha=1.0)
            ax[1].set_title("{} Q(s,a) vs. Observed Partial Returns".format(cumulant_name), pad=0)

            # Plot STD of Ensemble GVF values
            cumulant_name_std_traces = std_traces[alert_type][cumulant_name][:min_steps]
            thresh_list = [std_trace_thresholds[alert_type][cumulant_name] for _ in range(min_steps)]
            ax[2].plot(time_steps, thresh_list, c="k", label="STD Threshold", alpha=1.0)
            q_label = f'{cumulant_name} Q(s,a) STD'
            ax[2].plot(time_steps, cumulant_name_std_traces, label=q_label, c="cyan", alpha=1.0)
            ax[2].set_title(q_label, pad=0)
            ax[2].set_yscale("log")

            # Plot alert trace
            cumulant_name_traces = alert_traces[alert_type][cumulant_name][:min_steps]
            thresh_list = [alert_trace_thresholds[alert_type][cumulant_name] for _ in range(min_steps)]
            ax[3].plot(time_steps, thresh_list, c="k", label="Trace Threshold", alpha=1.0)
            ax[3].plot(time_steps, cumulant_name_traces, label="{} Trace".format(cumulant_name), c="m", alpha=1.0)
            ax[3].set_title(f"{cumulant_name} Q(s,a) vs. Observed Partial Returns Absolute Difference Trace", pad=0)

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            ax[3].legend()
            plt.xlabel("Time Step")
            fig.savefig(path / "{}_Uncertainty_Alerts_Summary_Plot.png".format(cumulant_name))
            plt.close()

    # Plot Composite Alert
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 12))

    # Alerts
    for step in time_steps:
        if composite_alerts[step]:
            ax.axvline(x=step, color='r', alpha=0.2)

    # Plot actions and endogenous variables
    for action_name in actions_taken:
        action_type_actions = actions_taken[action_name][:min_steps]
        ax.plot(time_steps, action_type_actions, label=action_name, alpha=1.0)
    for endo_obs_name in endo_obs:
        obs_type_observations = endo_obs[endo_obs_name][:min_steps]
        ax.plot(time_steps, obs_type_observations, label=endo_obs_name, alpha=1.0)
    ax.set_title("Actions and Endogenous Variables", pad=0)

    ax.legend()
    plt.xlabel("Time Step")
    fig.savefig(path / "Composite_Alert_Summary_Plot.png")
    plt.close()

def make_ensemble_info_step_plot(ensemble_info, iteration, path):
    # Make Max-Min Diff Histograms
    bins = [0.0, 0.01, 0.1, 1.0, 10.0, 1000.0, 100000.0]
    labels = ["0.0-0.01", "0.01-0.1", "0.1-1.0", "1.0-10.0", "1e1-1e3", "1e3-1e5"]
    # Make STD Histograms
    for alert_type in ensemble_info:
        for cumulant_name in ensemble_info[alert_type]:
            counts, np_bins = np.histogram(ensemble_info[alert_type][cumulant_name]["std"], bins=bins)
            plt.bar(labels, counts)
            plt.xlabel("STD")
            plt.ylabel("Counts")
            plt.savefig(path / "{}_Alert_Ensemble_STD_Histogram_Iteration_{}.png".format(cumulant_name, iteration))
            plt.close()

def make_ensemble_info_summary_plots(stats, path, prefix):
    stds = stats["training_stds"]

    for alert_type in stds:
        for cumulant_name in stds[alert_type]:
            x_steps = list(range(len(stds[alert_type][cumulant_name])))
            plt.plot(x_steps, stds[alert_type][cumulant_name], label="{} Normalized STD".format(cumulant_name))
            plt.xlabel("Training Iteration")
            plt.yscale("log")
            plt.legend()
            plt.savefig(path / "{}_Alert_Ensemble_{}_Summary.png".format(cumulant_name, prefix))
            plt.close()

def make_saturation_actor_critic_plot(
    states,
    actions,
    bootstrap_q_values,
    policy_q_values,
    ensemble_q_values,
    actor_params,
    env,
    path,
    prefix,
    epoch,
):
    # Currently assuming 'anytime_multi_trace' state constructor
    ensemble = len(ensemble_q_values)
    for i in range(0, len(states)):
        curr_state = states[i]
        curr_alpha = actor_params[i][0]
        curr_beta = actor_params[i][1]

        ax_title = 'Saturation: '
        for j in [2, 14, 12, 10, 8, 6, 4]:
            ax_title += "{:.1e} ".format(curr_state[j])

        ax_title += '\nAction: '
        for j in [1, 13, 11, 9, 7, 5, 3]:
            ax_title += "{:.1e} ".format(curr_state[j])

        ax_title += '\nAnytime Thermometer: '
        for j in [16, 17, 18, 19, 20, 21, 22]:
            ax_title += "{} ".format(curr_state[j])

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 12))

        beta_pdf = beta.pdf(actions, curr_alpha, curr_beta)
        beta_max = actions[beta_pdf.argmax()]
        #beta_max = beta.mean(curr_alpha, curr_beta)
        bootstrap_reduct_max = actions[bootstrap_q_values[i].argmax()]
        policy_reduct_max = actions[policy_q_values[i].argmax()]

        ax[0].axvline(x=beta_max, color='r', linestyle="dashed", alpha=1.0)
        ax[1].axvline(x=beta_max, color='r', linestyle="dashed", alpha=1.0)
        ax[0].axvline(x=bootstrap_reduct_max, color='b', linestyle="dashed", alpha=1.0)
        ax[1].axvline(x=bootstrap_reduct_max, color='b', linestyle="dashed", alpha=1.0)
        ax[0].axvline(x=policy_reduct_max, color='g', linestyle="dashed", alpha=1.0)
        ax[1].axvline(x=policy_reduct_max, color='g', linestyle="dashed", alpha=1.0)

        # Plot Actor Policy
        ax[0].plot(
                actions,
                beta_pdf,
                label="Actor Policy",
                c="r",
            )
        ax[0].set_title("Actor Policy", pad=0)

        # Plot each critic in the ensemble
        ax[1].plot(actions, bootstrap_q_values[i], label="Bootstrap Reduct", c="b", alpha=1.0)
        ax[1].plot(actions, policy_q_values[i], label="Policy Reduct", c="g", alpha=1.0)
        for j in range(ensemble):
            ax[1].plot(actions, ensemble_q_values[j][i], alpha=0.2)

        ax[1].set_title("Q-Function", pad=20)
        plt.xlabel("Action Space")
        fig.suptitle(ax_title)
        #ax.legend()
        fig.savefig(path / "{}_epoch_{}_state_{}_Summary_Plots.png".format(prefix, epoch, i))
        plt.close()

def make_reseau_actor_critic_plot(
    states,
    actions,
    bootstrap_q_values,
    policy_q_values,
    ensemble_q_values,
    actor_params,
    env,
    path,
    prefix,
    epoch,
):
    # Currently assuming 'ReseauAnytime' state constructor
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    mins = [0, action_space_low[0], obs_space_low[0], obs_space_low[1], action_space_low[0], action_space_low[0], action_space_low[0], action_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[1], obs_space_low[1], obs_space_low[1], obs_space_low[1], 0, 0] # noqa: E501 - lol
    maxs = [1, action_space_high[0], obs_space_high[0], obs_space_high[1], action_space_high[0], action_space_high[0], action_space_high[0], action_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[1], obs_space_high[1], obs_space_high[1], obs_space_high[1], 1, 1] # noqa: E501 - lol
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

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 12))

        beta_pdf = beta.pdf(actions, curr_alpha, curr_beta)
        beta_max = actions[beta_pdf.argmax()]
        #beta_max = beta.mean(curr_alpha, curr_beta)
        bootstrap_reduct_max = actions[bootstrap_q_values[i].argmax()]
        policy_reduct_max = actions[policy_q_values[i].argmax()]

        ax[0].axvline(x=beta_max, color='r', linestyle="dashed", alpha=1.0)
        ax[1].axvline(x=beta_max, color='r', linestyle="dashed", alpha=1.0)
        ax[0].axvline(x=bootstrap_reduct_max, color='b', linestyle="dashed", alpha=1.0)
        ax[1].axvline(x=bootstrap_reduct_max, color='b', linestyle="dashed", alpha=1.0)
        ax[0].axvline(x=policy_reduct_max, color='g', linestyle="dashed", alpha=1.0)
        ax[1].axvline(x=policy_reduct_max, color='g', linestyle="dashed", alpha=1.0)

        # Plot Actor Policy
        ax[0].plot(
                actions,
                beta_pdf,
                label="Actor Policy",
                c="r",
            )
        ax[0].set_title("Actor Policy", pad=0)

        # Plot each critic in the ensemble
        ax[1].plot(actions, bootstrap_q_values[i], label="Bootstrap Reduct", c="b", alpha=1.0)
        ax[1].plot(actions, policy_q_values[i], label="Policy Reduct", c="g", alpha=1.0)
        for j in range(ensemble):
            ax[1].plot(actions, ensemble_q_values[j][i], alpha=0.2)

        ax[1].set_title("Q-Function", pad=20)
        plt.xlabel("Action Space")
        fig.suptitle(ax_title)
        ax.legend()
        fig.savefig(path / "{}_epoch_{}_test_state_{}_Summary_Plots.png".format(prefix, epoch, i))
        plt.close()

def make_reseau_gvf_critic_plot(plot_info, env, path, prefix, epoch):
    # Currently assuming 'ReseauAnytime' state constructor
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    mins = [0, action_space_low[0], obs_space_low[0], obs_space_low[1], action_space_low[0], action_space_low[0], action_space_low[0], action_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[0], obs_space_low[1], obs_space_low[1], obs_space_low[1], obs_space_low[1], 0, 0] # noqa: E501 - lol
    maxs = [1, action_space_high[0], obs_space_high[0], obs_space_high[1], action_space_high[0], action_space_high[0], action_space_high[0], action_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[0], obs_space_high[1], obs_space_high[1], obs_space_high[1], obs_space_high[1], 1, 1] # noqa: E501 - lol
    states = plot_info["states"]
    actions = plot_info["actions"]
    for i in range(0, len(states)):
        curr_state = states[i]

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

        for cumulant_name in plot_info["q_values"]:
            ensemble = len(plot_info["ensemble_qs"][cumulant_name])
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot each critic in the ensemble
            ax.plot(actions, plot_info["q_values"][cumulant_name][i], label="Q Function", c="b", alpha=1.0)
            for j in range(ensemble):
                ax.plot(actions, plot_info["ensemble_qs"][cumulant_name][j][i], alpha=0.2)

            # ax.set_title("{} Q-Function".format(cumulant_name), pad=0)
            plt.xlabel("Action Space")
            fig.suptitle(ax_title)
            # ax.legend()
            fig.savefig(path / "{}_{}_epoch_{}_state_{}_Critic_Ensemble.png".format(prefix, cumulant_name, epoch, i))
            plt.close()

def radar(ax, normalizations, df, *, id_column, title=None, max_values=None, padding=1.25):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()

    # if max_values is None:
    #     max_values = {key: padding * max(value) for key, value in data.items()}
    # normalized_data = {key: np.array(value) / max_values[key] for key, value in data.items()}
    # normalized_data = {key: (np.array(value) - normalizations.get(key, [0, 1])[0]) / (normalizations.get(key, [0, 1])[1] - normalizations.get(key, [0, 1])[0]) for key, value in data.items()}
    normalized_data = {}
    for key, value in data.items():
        #first < second: larger the better
        #first > second: smaller the better
        first = normalizations.get(key, [0, 1])[0]
        second = normalizations.get(key, [0, 1])[1]
        normalized_data[key] = (np.array(value) - first) / (second - first)

    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.15)
        for _x, _y, t, at in zip(angles, values, values, actual_values):
            _y = 0.9
            t = f'{t:.2f}\n({at:.2f})' if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, fontsize=7, va='center', ha='center')

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=7)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    if title is not None: ax.set_title(title)

def make_summary_plot(plot_info: dict, path: Path, prefix: str):
    formal_labels = {
        'bellman_error': 'BE',
        'ibe_smoothed_change': 'BEC',
        'td_error': 'TDE',
        'action_gap': 'Pi Diff', # L2 distance between greedy action and learned action
        'greedy_gaps': 'Q Gap', # Gap between q estimation of greedy action and learned action
        'policy_improvements': 'Q Imprv', # q estimation of learned action minus q estimation of dataset action
        'q_estimation_avg': 'Avg Q',
        'q_estimation_median': 'Med Q',
        'q_estimation_min': 'Min Q',
        'q_estimation_max': 'Max Q',
        'ensemble_ste': 'Ens STE',
        'local_max': 'Local Max',
        'local_min': 'Local Min',
        'linear': 'Linear',
        'flat': 'Flat',
        'num_episodes': '#Ep',
        'rewards': 'R',
        'avg_reward': 'Avg R',
        'avg_return': 'Avg G',
        'composite_alerts': 'Alerts',
        'efficiency_predictions': 'Eff Pred',
    }
    larger_better_list = [
        'x',
        'Q Imprv',
        'Local Max',
        'R',
        'Avg R',
        'Avg G',
        'Eff Pred',
    ]
    smaller_better_list = [
        'x',
        'Pi Diff',
        'BEC',
        'TDE',
        'Q Gap',
        'Local Min',
        'Linear',
        'Flat',
    ]
    statistic_list = [
        'x',
        'BE',
        'Avg Q',
        'Med Q',
        'Min Q',
        'Max Q',
        '#Ep',
        'Alerts',
        'Ens STE',
    ]
    normalizations = {
        'BE': [0, 1],
        'BEC': [0, 1],
        'TDE': [0, 1],
        'Q Gap': [-1, 1],
        'Q Imprv': [-1, 1],
        'Pi Gap': [-1, 1],
        'Avg Q': [0, 10],
        'Med Q': [0, 10],
        'Min Q': [0, 10],
        'Max Q': [0, 10],
        'Ens STE': [0, 10],
        'Local Max': [0, 1],
        'Local Min': [0, 1],
        'Linear': [0, 1],
        'Flat': [0, 1],
        '#Ep': [0, 1000],
        'R': [0, 2],
        'Avg R': [0, 2],
        'Avg G': [0, 1000],
        'Alerts': [0, 1]
    }
    order = {
        'bellman_error': 0,
        'ibe_smoothed_change': 0,
        'td_error': 0,
        'action_gap': 1,
        'greedy_gaps': 1,
        'policy_improvements': 1,
        'q_estimation_avg': 2,
        'q_estimation_median': 2,
        'q_estimation_min': 2,
        'q_estimation_max': 2,
        'ensemble_ste': 2,
        'local_max': 3,
        'local_min': 3,
        'linear': 3,
        'flat': 3,
        'num_episodes': 4,
        'rewards': 4,
        'avg_reward': 4,
        'avg_reward (1)': 4,
        'avg_return': 4,
        'composite_alerts': 5,
    }
    plot_info_keys = list(plot_info.keys())
    plot_info_orders = [order.get(k, -1) for k in plot_info_keys]
    ordered_keys = [x for _,x in sorted(zip(plot_info_orders, plot_info_keys))]
    data = {'x': [*'0']}
    for k in ordered_keys:
        v = plot_info[k]
        if type(v) == list and len(v) > 0:
            v = v[-10:]
            if type(v) == np.ndarray or type(v) == list: # ensemble size > 1
                v = np.asarray(v).mean()
        elif type(v) == list and len(v) == 0:
            v = None
        data[formal_labels.get(k, k)] = [v]

    larger_better = {}
    for k in data.keys():
        if k in larger_better_list:
            larger_better[k] = data[k]
    smaller_better = {}
    for k in data.keys():
        if k in smaller_better_list:
            smaller_better[k] = data[k]
    statistic_only = {}
    for k in data.keys():
        if k in statistic_list:
            statistic_only[k] = data[k]

    fig, ax = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(polar=True))
    radar(
        ax[0], normalizations,
        pd.DataFrame(larger_better),
        id_column='x',
        title='Larger the better',
        padding=1.1
    )
    radar(
        ax[1], normalizations,
        pd.DataFrame(smaller_better),
        id_column='x',
        title='Closer to 0 the better',
        padding=1.1
    )
    radar(
        ax[2], normalizations,
        pd.DataFrame(statistic_only),
        id_column='x',
        title='Statistic',
        padding=1.1
    )
    plt.tight_layout()
    fig.savefig(path / "{}_summary.png".format(prefix), dpi=300)
    plt.close()
    plt.clf()



def make_online_plots(freezer, stats, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    #make_trace_alerts_plots(stats, save_path)
    #make_uncertainty_alerts_plots(stats, save_path)
    #make_ensemble_info_summary_plots(stats, save_path, "Online")
    """
    make_action_mean_variance_plot(freezer, save_path)
    make_action_gap_plot(stats, save_path)
    make_bellman_error_plot(stats, save_path, "online")
    """
    #make_param_plot(freezer, save_path)
    #make_reward_plot(stats, save_path)
    #make_cumulative_reward_plot(stats, save_path)

def make_offline_plots(freezer, stats, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    make_bellman_error_plot(stats, save_path, "offline")
    make_td_error_plot(stats, save_path, "offline")
    make_action_gap_plot(stats, save_path, "offline")
    make_q_estimation_plot(stats, save_path, "offline")
    make_policy_improvement_plot(stats, save_path, "offline")
    # make_train_test_loss_plot(stats, save_path)
    make_counterfactual_plot(stats, save_path, "offline")
    make_summary_plot(stats, save_path, "offline")

def make_actor_critic_plots(agent, env, plot_transitions, prefix, iteration, save_path):
    res = get_test_state_qs_and_policy_params(agent, plot_transitions) # noqa
    #make_reseau_actor_critic_plot(*res, env, save_path, prefix, iteration)
    #make_saturation_actor_critic_plot(*res, env, save_path, prefix, iteration)
