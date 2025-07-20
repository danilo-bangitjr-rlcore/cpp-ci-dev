from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import corerl.interaction.checkpointing as chk


class DummyCheckpointable:
    def __init__(self):
        self.saved = []
        self.loaded = []
    def save(self, path: Path):
        self.saved.append(path)
    def load(self, path: Path):
        self.loaded.append(path)

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
    cliff = now - timedelta(hours=1)
    freq = timedelta(minutes=10)

    # 5 checkpoints, 10min apart
    times = [now - timedelta(minutes=10*i) for i in range(5)][::-1]
    paths = [Path(f"chk_{i}") for i in range(5)]

    # Should keep first and last, and those after cliff
    to_delete = chk.prune_checkpoints(paths, times, cliff, freq)

    assert all(isinstance(p, Path) for p in to_delete)
    assert paths[0] not in to_delete
    assert paths[-1] not in to_delete

def test_checkpoint_and_restore(tmp_path: Path):
    """
    Test that checkpoint creates a directory and calls save, and restore_checkpoint loads the latest checkpoint.
    TODO: replace this test with real operations instead of mocks.
    """
    cfg = MagicMock()
    cfg.checkpoint_path = tmp_path
    cfg.restore_checkpoint = True

    agent = DummyCheckpointable()
    app_state = DummyCheckpointable()

    now = datetime(2023, 1, 1, 12, 0, 0)
    cliff = timedelta(hours=1)
    freq = timedelta(minutes=10)

    # Test checkpoint creates dir and calls save
    last = now - timedelta(hours=2)
    chk.checkpoint(now, cfg, agent, app_state, last, cliff, freq)

    assert agent.saved
    assert app_state.saved

    # Test restore loads latest (use the directory created by checkpoint)
    chk.restore_checkpoint(cfg, agent, app_state)
    assert agent.loaded
    assert app_state.loaded
