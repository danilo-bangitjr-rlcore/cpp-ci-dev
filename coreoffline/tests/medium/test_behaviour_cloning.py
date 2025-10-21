from corerl.state import AppState
from lib_agent.buffer.datatypes import Trajectory

from coreoffline.scripts.behaviour_clone import run_behaviour_cloning
from coreoffline.utils.config import OfflineMainConfig


def test_run_behaviour_cloning_smoke_test(
    dummy_app_state: AppState,
    offline_cfg: OfflineMainConfig,
    trajectories_with_timestamps: list[Trajectory],
):
    dummy_app_state.cfg = offline_cfg
    dummy_app_state.cfg.behaviour_clone.k_folds = 2
    dummy_app_state.cfg.behaviour_clone.mlp.epochs = 1
    run_behaviour_cloning(dummy_app_state, trajectories_with_timestamps)
