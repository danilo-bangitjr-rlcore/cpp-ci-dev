from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from corerl.state import AppState


@dataclass
class MetricsPlottingConfig:
    app_state: AppState
    step_start: int | None = None
    step_end: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    labels: list[str] | None = None

def make_mc_eval_plot(
    cfg: MetricsPlottingConfig,
    save_path: Path,
):
    if not cfg.app_state.cfg.eval_cfgs.monte_carlo.enabled:
        return

    assert cfg.labels is not None

    for label in cfg.labels:
        state_v_df = cfg.app_state.metrics.read(
            metric=f"state_v_{label}",
            step_start=cfg.step_start,
            step_end=cfg.step_end,
            start_time=cfg.start_time,
            end_time=cfg.end_time
        )
        observed_a_q_df = cfg.app_state.metrics.read(
            metric=f"observed_a_q_{label}",
            step_start=cfg.step_start,
            step_end=cfg.step_end,
            start_time=cfg.start_time,
            end_time=cfg.end_time
        )
        partial_return_df = cfg.app_state.metrics.read(
            metric=f"partial_return_{label}",
            step_start=cfg.step_start,
            step_end=cfg.step_end,
            start_time=cfg.start_time,
            end_time=cfg.end_time
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
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=None))
            plt.gcf().autofmt_xdate()

        plt.ylabel("Return")
        plt.xlabel("Time")
        plt.legend()
        plt.savefig(save_path / f"monte_carlo_eval_{label}.png")
        plt.close()

def plot_metrics(
    cfg: MetricsPlottingConfig
):
    if cfg.labels is None or len(cfg.labels) == 0:
        cfg.labels = [""]

    save_path = Path(cfg.app_state.cfg.experiment.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    make_mc_eval_plot(cfg, save_path)
