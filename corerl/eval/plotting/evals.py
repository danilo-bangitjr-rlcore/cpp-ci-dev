from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from corerl.data_pipeline.tag_config import TagConfig
from corerl.eval.actor_critic import Action, State
from corerl.state import AppState


def make_tag_title(tag_name: str, col_names: list[str], values: np.ndarray) -> str:
    """
    Get the state/action info that pertains to a given tag
    """
    inds = []
    for i in range(len(col_names)):
        if tag_name in col_names[i]:
            inds.append(i)
    tag_vals = [f"{float(val):.3f}" for val in values[inds]]
    title = ""
    if len(inds) > 0:
        title = f"{tag_name}: {tag_vals}\n"

    return title

def make_state_info_title(
    tags: list[TagConfig],
    state_cols: list[str],
    action_cols: list[str],
    state: list[float],
    action: list[float],
):
    """
    Produce a plot title consisting of rows for each tag in 'tags' that has a state constructor or action constructor.
    Each row consists of all the state/action features produced for the given tag
    """
    state_title = "State Info:\n"
    action_title = "Current Action:\n"
    state_np = np.array(state)
    action_np = np.array(action)
    for tag in tags:
        tag_name = tag.name
        state_title += make_tag_title(tag_name, state_cols, state_np)
        action_title += make_tag_title(tag_name, action_cols, action_np)

    title = state_title + action_title

    return title

def make_actor_critic_plots(
    app_state: AppState,
    save_path: Path,
    labels: list[str],
    step_start: int | None = None,
    step_end: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None
):
    """
    Plot for the Actor-Critic evaluator.
    Top subplot represents the probability density function of the policy along a given action dim at the given state.
    Bottom subplot represents the critic at the given state along the full direct action range for a given action dim.
    In the delta action case, a middle subplot is produced that displays the critic over the delta action range
    """
    if not app_state.cfg.eval_cfgs.actor_critic.enabled:
        return

    tags = app_state.cfg.pipeline.tags
    for label in labels:
        # Get entry from evals table in TSDB
        ac_eval_df = app_state.evals.read(
            evaluator=f"actor-critic_{label}",
            step_start=step_start,
            step_end=step_end,
            start_time=start_time,
            end_time=end_time
        )

        qs_and_policy = ac_eval_df["value"].iloc[0]
        state_cols = qs_and_policy["state_cols"]
        state_cols = cast(list[str], state_cols)
        action_cols = qs_and_policy["action_cols"]
        action_cols = cast(list[str], action_cols)

        # Produce plot for each action dimension in each test state
        state_counter = 0
        for state in qs_and_policy["plot_info"]:
            for action in qs_and_policy["plot_info"][state]["a_dim"]:
                if app_state.cfg.agent.policy_manager.delta_action:
                    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

                    # Plot critic over delta action range
                    delta_action_range = qs_and_policy["plot_info"][state]["a_dim"][action]["delta_critic_actions"]
                    for critic in qs_and_policy["plot_info"][state]["a_dim"][action]["delta_critic"]:
                        axs[1].plot(delta_action_range, critic)

                    axs[1].set_title("Critic Over Delta Action Range")
                    axs[1].set_xlabel(action)
                    axs[1].set_ylabel("Q")
                else:
                    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

                # Plot policy
                policy_actions = qs_and_policy["plot_info"][state]["a_dim"][action]["policy_actions"]
                for pdf in qs_and_policy["plot_info"][state]["a_dim"][action]["pdf"]:
                    axs[0].plot(policy_actions, pdf)

                axs[0].set_title("Policy")
                axs[0].set_xlabel(action)
                axs[0].set_ylabel("Probability Density")

                # Plot critic over direct action range
                full_action_range = qs_and_policy["plot_info"][state]["a_dim"][action]["direct_critic_actions"]
                for critic in qs_and_policy["plot_info"][state]["a_dim"][action]["direct_critic"]:
                    axs[-1].plot(full_action_range, critic)

                axs[-1].set_title("Critic Over Direct Action Range")
                axs[-1].set_xlabel(action)
                axs[-1].set_ylabel("Q")

                # Format state info in plot title
                state_l = State.model_validate_json(state).state
                action_json = qs_and_policy["plot_info"][state]["current_action"]
                current_action_l = Action.model_validate_json(action_json).action
                title = make_state_info_title(tags, state_cols, action_cols, state_l, current_action_l)
                fig.suptitle(title, fontsize=12)
                fig.tight_layout()

                fig.savefig(save_path / f"iter_{label}_state_{state_counter}_varying_{action}_actor_critic_plot.png")
                plt.close(fig)

            state_counter += 1

def plot_evals(
    app_state: AppState,
    step_start: int | None = None,
    step_end: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    labels: list[str] | None = None,
):
    if labels is None or len(labels) == 0:
        labels = [""]

    save_path = Path(app_state.cfg.experiment.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    make_actor_critic_plots(app_state, save_path, labels, None, None, None, None)
