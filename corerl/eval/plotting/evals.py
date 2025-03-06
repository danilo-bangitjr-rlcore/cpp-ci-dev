from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from corerl.data_pipeline.tag_config import TagConfig
from corerl.eval.actor_critic import PlotInfoBatch
from corerl.state import AppState


@dataclass
class EvalsPlottingConfig:
    app_state: AppState
    step_start: int | None = None
    step_end: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    labels: list[str] | None = None

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
    cfg: EvalsPlottingConfig,
    save_path: Path,
):
    """
    Plot for the Actor-Critic evaluator.
    Top subplot represents the probability density function of the policy along a given action dim at the given state.
    Bottom subplot represents the critic at the given state along the full direct action range for a given action dim.
    In the delta action case, a middle subplot is produced that displays the critic over the delta action range
    """
    if not cfg.app_state.cfg.eval_cfgs.actor_critic.enabled:
        return

    assert cfg.labels is not None

    tags = cfg.app_state.cfg.pipeline.tags
    for label in cfg.labels:
        # Get entry from evals table in TSDB
        ac_eval_df = cfg.app_state.evals.read(
            evaluator=f"actor-critic_{label}",
            step_start=cfg.step_start,
            step_end=cfg.step_end,
            start_time=cfg.start_time,
            end_time=cfg.end_time
        )
        plot_info_batch = PlotInfoBatch.model_validate_json(ac_eval_df["value"].iloc[0])
        state_cols = plot_info_batch.state_cols
        action_cols = plot_info_batch.action_cols

        # Produce actor-critic plot for each action dim at each test state
        state_count = 0
        for state_plot_info in plot_info_batch.states:
            state = state_plot_info.state
            action = state_plot_info.current_action
            for action_plot_info in state_plot_info.a_dims:
                action_tag = action_plot_info.action_tag
                if cfg.app_state.cfg.agent.policy.delta_actions:
                    assert action_plot_info.delta_critic is not None
                    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

                    # Plot critic over delta action range
                    delta_x_range = action_plot_info.delta_critic.x_range
                    for q_func in action_plot_info.delta_critic.q_vals:
                        axs[1].plot(delta_x_range, q_func)

                    axs[1].set_title("Critic Over Delta Action Range")
                    axs[1].set_xlabel(action_tag)
                    axs[1].set_ylabel("Q")
                else:
                    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

                # Plot policy
                policy_x_range = action_plot_info.pdf.x_range
                for pdf in action_plot_info.pdf.pdfs:
                    axs[0].plot(policy_x_range, pdf)

                loc = action_plot_info.pdf.loc
                scale = action_plot_info.pdf.scale
                axs[0].set_title(f"Policy - Loc: {loc:.3f}, Scale: {scale:.3f}")
                axs[0].set_xlabel(action_tag)
                axs[0].set_ylabel("Probability Density")

                # Plot critic over direct action range
                direct_x_range = action_plot_info.direct_critic.x_range
                for q_func in action_plot_info.direct_critic.q_vals:
                    axs[-1].plot(direct_x_range, q_func)

                axs[-1].set_title("Critic Over Direct Action Range")
                axs[-1].set_xlabel(action_tag)
                axs[-1].set_ylabel("Q")

                # Plot current a_dim value
                for ax in axs:
                    ax.axvline(action_plot_info.action_val, color='r', linestyle='--', label="Current Action")
                    ax.legend()

                # Format state info in plot title
                title = make_state_info_title(tags, state_cols, action_cols, state, action)
                fig.suptitle(title, fontsize=12)
                fig.tight_layout()

                fig.savefig(save_path / f"iter_{label}_state_{state_count}_varying_{action_tag}_actor_critic_plot.png")
                plt.close(fig)

            state_count += 1

def plot_evals(cfg: EvalsPlottingConfig):
    if cfg.labels is None or len(cfg.labels) == 0:
        cfg.labels = [""]

    save_path = Path(cfg.app_state.cfg.experiment.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    make_actor_critic_plots(cfg, save_path)