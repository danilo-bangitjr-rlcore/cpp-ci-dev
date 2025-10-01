from corerl.data_pipeline.datatypes import Transition
from corerl.state import AppState

from coreoffline.behaviour_cloning.main import run_behaviour_cloning
from coreoffline.config import OfflineMainConfig


def test_run_behaviour_cloning_smoke_test(
    dummy_app_state: AppState,
    offline_cfg: OfflineMainConfig,
    transitions_with_timestamps: list[Transition],
):
    dummy_app_state.cfg = offline_cfg
    dummy_app_state.cfg.behaviour_clone.k_folds = 2
    dummy_app_state.cfg.behaviour_clone.mlp.epochs = 1
    run_behaviour_cloning(dummy_app_state, transitions_with_timestamps)
