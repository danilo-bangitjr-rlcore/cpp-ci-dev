from datetime import datetime, timedelta
from pathlib import Path

import corerl.interaction.checkpointing as chk
from corerl.agent.greedy_ac import GreedyAC
from corerl.configs.interaction.config import InteractionConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState


def test_next_power_of_2():
    """
    Test that next_power_of_2 returns the correct next power of two for various inputs.
    """
    assert chk.next_power_of_2(0) == 1
    assert chk.next_power_of_2(1) == 1
    assert chk.next_power_of_2(2) == 2
    assert chk.next_power_of_2(3) == 4
    assert chk.next_power_of_2(7) == 8
    assert chk.next_power_of_2(16) == 16

def test_prev_power_of_2():
    """
    Test that prev_power_of_2 returns the correct previous power of two for various inputs.
    """
    assert chk.prev_power_of_2(0) == 1
    assert chk.prev_power_of_2(1) == 1
    assert chk.prev_power_of_2(2) == 2
    assert chk.prev_power_of_2(3) == 2
    assert chk.prev_power_of_2(7) == 4
    assert chk.prev_power_of_2(16) == 16

def test_periods_since():
    """
    Test that periods_since computes the correct number of periods between two datetimes.
    """
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 1, 1, 0, 0)
    period = timedelta(minutes=10)
    assert chk.periods_since(start, end, period) == 6

def test_prune_checkpoints_basic():
    """
    Test that prune_checkpoints returns the correct checkpoints to delete, keeping first, last, and recent ones.
    """
    now = datetime(2023, 1, 1, 12, 0, 0)
    cliff = now - timedelta(minutes=20)
    freq = timedelta(minutes=10)

    # 8 checkpoints, 10min apart
    times = [now - freq * i for i in range(8)][::-1]
    paths = [Path(f"chk_{i}") for i in range(8)]

    # Should keep first and last, and those after cliff
    to_delete = chk.prune_checkpoints(paths, times, cliff, freq)

    assert paths[0] not in to_delete
    assert paths[-1] not in to_delete
    assert to_delete == [Path('chk_2')]

def test_checkpoint_and_restore(tmp_path: Path, dummy_app_state: AppState):
    """
    Test that checkpoint creates a directory and calls save, and restore_checkpoint loads the latest checkpoint
    using a real agent and a real app_state.
    """
    # Setup config for checkpointing
    interaction_cfg = InteractionConfig(
        checkpoint_path=tmp_path,
        restore_checkpoint=True,
        checkpoint_freq=timedelta(minutes=10),
        checkpoint_cliff=timedelta(hours=1),
        obs_period=timedelta(seconds=1),
        action_period=timedelta(seconds=1),
        state_age_tol=timedelta(seconds=1),
        load_historical_data=False,
        historical_windows=[],
        historical_batch_size=1,
        update_warmup=0,
        write_obs_to_csv=False,
        warmup_period=None,
    )

    # Setup a real agent
    pipeline = Pipeline(dummy_app_state, dummy_app_state.cfg.pipeline)
    column_desc = pipeline.column_descriptions
    agent = GreedyAC(
        dummy_app_state.cfg.agent,
        dummy_app_state,
        column_desc,
    )

    now = datetime(2023, 1, 1, 12, 0, 0)
    last = now - timedelta(hours=2)
    cliff = timedelta(hours=1)
    freq = timedelta(minutes=10)

    # Save checkpoint
    chk.checkpoint(
        now,
        interaction_cfg,
        last,
        cliff,
        freq,
        elements=(agent, dummy_app_state),
    )

    # Check that checkpoint directory exists
    checkpoint_dirs = list(tmp_path.glob("*"))
    assert checkpoint_dirs, "Checkpoint directory was not created."

    # Load checkpoint
    chk.restore_checkpoint(
        interaction_cfg,
        elements=(agent, dummy_app_state),
    )
