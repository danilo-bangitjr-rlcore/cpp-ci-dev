import time
from datetime import UTC, datetime, timedelta

import pytest

from corerl.utils.app_time import AppTime


def test_demo_time_computation():
    """Test synthetic time computation for demo mode."""
    start_time = datetime(2020, 1, 1, tzinfo=UTC)
    obs_period = timedelta(minutes=5)

    app_time = AppTime(
        is_demo=True,
        start_time=start_time,
        obs_period=obs_period,
    )

    # Step 0
    assert app_time.get_current_time() == datetime(2020, 1, 1, 0, 0, tzinfo=UTC)

    # Step 1
    app_time.increment_step()
    assert app_time.get_current_time() == datetime(2020, 1, 1, 0, 5, tzinfo=UTC)

    # Step 2
    app_time.increment_step()
    assert app_time.get_current_time() == datetime(2020, 1, 1, 0, 10, tzinfo=UTC)


@pytest.fixture
def normal_app_time():
    return AppTime(
        is_demo=False,
        start_time=datetime.now(UTC),
    )


def test_normal_mode_uses_wall_clock(normal_app_time: AppTime):
    """Test that normal mode uses real wall-clock time."""
    current = normal_app_time.get_current_time()
    time.sleep(0.01)
    after = datetime.now(UTC)
    assert current <= after


def test_normal_mode_increment_step_doesnt_affect_time(normal_app_time: AppTime):
    """Test that increment_step doesn't affect normal mode time."""
    time1 = normal_app_time.get_current_time()
    normal_app_time.increment_step()
    time2 = normal_app_time.get_current_time()

    # Times should be very close (both call datetime.now)
    assert abs((time2 - time1).total_seconds()) < 1.0


def test_getstate_setstate():
    """Test __getstate__ and __setstate__ methods."""
    original = AppTime(
        is_demo=True,
        start_time=datetime(2020, 1, 1, tzinfo=UTC),
        obs_period=timedelta(hours=1),
        agent_step=100,
    )

    state = original.__getstate__()

    # Verify state contains all fields
    assert state['agent_step'] == 100
    assert state['start_time'] == datetime(2020, 1, 1, tzinfo=UTC)
    assert state['is_demo'] is True
    assert state['obs_period'] == timedelta(hours=1)

    # Create new instance and restore
    restored = AppTime(
        is_demo=False,
        start_time=datetime.now(UTC),
    )
    restored.__setstate__(state)

    # Verify restored matches original
    assert restored.agent_step == original.agent_step
    assert restored.start_time == original.start_time
    assert restored.is_demo == original.is_demo
    assert restored.obs_period == original.obs_period
