import json
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from corerl.config import MainConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.state import AppState


def make_state_info_title(tags: list[TagConfig], state_cols: list[str], state: list[float]):
    """
    Produce a plot title consisting of rows for each tag in 'tags' that has a state constructor.
    Each row consists of all the state features produced for the given tag
    """
    state_np = np.array(state)
    title = ""
    for tag in tags:
        tag_name = tag.name
        tag_inds = []
        for i in range(len(state_cols)):
            if tag_name in state_cols[i]:
                tag_inds.append(i)
        tag_vals = [f"{float(val):.3f}" for val in state_np[tag_inds]]
        title += f"{tag_name}: {tag_vals}\n"

    return title

def make_actor_critic_plots(
    cfg: MainConfig,
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
    Bottom subplot represents the critic at the given state along the same action dimension.
    """
    if not cfg.eval_cfgs.actor_critic.enabled:
        return

    tags = cfg.pipeline.tags
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

        state_counter = 0
        # Produce plot for each action dimension in each test state
        for state in qs_and_policy["states"]:
            for action in qs_and_policy["states"][state]:
                fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 15))
                x_range = qs_and_policy["states"][state][action]["actions"]
                # Plot policy
                for pdf in qs_and_policy["states"][state][action]["pdf"]:
                    axs[0].plot(x_range, pdf)

                axs[0].set_ylabel("Probability Density")

                # Plot critic
                for critic in qs_and_policy["states"][state][action]["critic"]:
                    axs[1].plot(x_range, critic)

                axs[1].set_xlabel(action)
                axs[1].set_ylabel("Q")

                # Format state info in plot title
                state_l: list[float] = json.loads(state)
                title = make_state_info_title(tags, state_cols, state_l)
                fig.suptitle(title, fontsize=12)
                fig.tight_layout()

                fig.savefig(save_path / f"iter_{label}_state_{state_counter}_varying_{action}_actor_critic_plot.png")
                plt.close(fig)

            state_counter += 1

def plot_evals(
    cfg: MainConfig,
    app_state: AppState,
    save_path: Path,
    step_start: int | None = None,
    step_end: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    labels: list[str] | None = None,
):
    if labels is None or len(labels) == 0:
        labels = [""]

    make_actor_critic_plots(cfg, app_state, save_path, labels, None, None, None, None)
