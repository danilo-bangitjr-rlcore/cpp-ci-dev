import sys
from datetime import UTC, datetime

from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.eval.plotting.evals import EvalsPlottingConfig, plot_evals
from corerl.messages.event_bus import EventBus
from corerl.state import AppState


def _get_cli_flags():
    flags: dict[str, str] = {}
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith("--step_start="):
            flags["step_start"] = arg.split("=")[1]
        if arg.startswith("--step_end="):
            flags["step_end"] = arg.split("=")[1]
        if arg.startswith("--start_time="):
            flags["start_time"] = arg.split("=")[1]
        if arg.startswith("--end_time="):
            flags["end_time"] = arg.split("=")[1]
        if arg.startswith("--labels="):
            flags["labels"] = arg.split("=")[1]

    return flags

def _create_plot_cfg(app_state: AppState) -> EvalsPlottingConfig:
    plot_cfg = EvalsPlottingConfig(app_state)
    flags = _get_cli_flags()
    if "step_start" in flags:
        plot_cfg.step_start = int(flags["step_start"])
    if "step_end" in flags:
        plot_cfg.step_end = int(flags["step_end"])
    if "start_time" in flags:
        start_time = datetime.strptime(flags["start_time"], "%m/%d/%Y %H:%M:%S")
        start_time = start_time.replace(tzinfo=UTC)
        plot_cfg.start_time = start_time
    if "end_time" in flags:
        end_time = datetime.strptime(flags["end_time"], "%m/%d/%Y %H:%M:%S")
        end_time = end_time.replace(tzinfo=UTC)
        plot_cfg.end_time = end_time
    if "labels" in flags:
        plot_cfg.labels = flags["labels"].split(" ")

    return plot_cfg

@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    event_bus = EventBus(cfg.event_bus, cfg.env)
    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=event_bus,
    )

    plot_cfg = _create_plot_cfg(app_state)
    plot_evals(plot_cfg)

if __name__ == "__main__":
    """
    Ex: python3 corerl/plot_evals.py --base=projects/victoria_ww/configs
    --config-name=offline_pretraining --labels="0 100"
    """
    main()
