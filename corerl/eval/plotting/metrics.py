from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from corerl.state import AppState


def make_mc_eval_plot(
    app_state: AppState,
    save_path: Path,
    labels: list[str],
    step_start: int | None = None,
    step_end: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None
):
    if not app_state.cfg.eval_cfgs.monte_carlo.enabled:
        return

    for label in labels:
        state_v_df = app_state.metrics.read(
            metric=f"state_v_{label}",
            step_start=step_start,
            step_end=step_end,
            start_time=start_time,
            end_time=end_time
        )
        observed_a_q_df = app_state.metrics.read(
            metric=f"observed_a_q_{label}",
            step_start=step_start,
            step_end=step_end,
            start_time=start_time,
            end_time=end_time
        )
        partial_return_df = app_state.metrics.read(
            metric=f"partial_return_{label}",
            step_start=step_start,
            step_end=step_end,
            start_time=start_time,
            end_time=end_time
        )

        if "time" in state_v_df:
            x_axis = state_v_df["time"].to_numpy()
        elif "agent_step" in state_v_df:
            x_axis = state_v_df["agent_step"].to_numpy()
        else:
            x_axis = list(range(len(state_v_df)))

        state_vs = state_v_df["value"].to_numpy()
        observed_a_qs = observed_a_q_df["value"].to_numpy()
        partial_returns = partial_return_df["value"].to_numpy()

        plt.scatter(x_axis, state_vs, s=8, alpha=0.25, label="Agent Q(s, a~pi(s))")
        plt.scatter(x_axis, observed_a_qs, s=8, alpha=0.25, label="Agent Q(s, observed_a)")
        plt.scatter(x_axis, partial_returns, s=8, alpha=0.25, label="Observed Return")

        if "time" in state_v_df:
            assert end_time is not None and start_time is not None
            interval = int(float((end_time - start_time).days) / 15.0) + 1
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            plt.gcf().autofmt_xdate()

        plt.ylabel("Return")
        plt.xlabel("Time")
        plt.legend()
        plt.savefig(save_path / f"monte_carlo_eval_{label}.png")
        plt.close()

def plot_metrics(
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

    make_mc_eval_plot(app_state, save_path, labels, step_start, step_end, start_time, end_time)
