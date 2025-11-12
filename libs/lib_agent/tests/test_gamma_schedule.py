import pytest

from lib_agent.gamma_schedule import LogarithmicGammaScheduler


@pytest.mark.parametrize(
    "max_gamma,horizon,update_interval",
    [
        (0.99, 1000, 100),
        (0.95, 500, 50),
        (0.999, 2000, 200),
        (0.9, 1000, 10),
    ],
)
def test_logarithmic_gamma_schedule_boundaries(
    max_gamma: float,
    horizon: int,
    update_interval: int,
):
    """Tests boundary conditions for logarithmic gamma scheduler."""
    scheduler = LogarithmicGammaScheduler(
        max_gamma=max_gamma,
        horizon=horizon,
        update_interval=update_interval,
    )

    # At step 0, gamma should be 0
    gamma_at_0 = scheduler.get_gamma(0)
    assert gamma_at_0 == 0, f"Expected gamma=0 at step 0, got {gamma_at_0}"

    gamma_at_horizon = scheduler.get_gamma(horizon)
    assert (
        abs(gamma_at_horizon - max_gamma) < 1e-6
    ), f"Expected gamma={max_gamma} at step {horizon}, got {gamma_at_horizon}"

    # At steps >= horizon, gamma should equal max_gamma
    for step_offset in [1, update_interval + 1, horizon + 1, horizon * 2 + 1]:
        step = horizon + step_offset
        # Skip if step is exactly on update_interval boundary
        if step % update_interval == 0:
            continue
        gamma = scheduler.get_gamma(step)
        assert (
            abs(gamma - max_gamma) < 1e-6
        ), f"Expected gamma={max_gamma} at step {step}, got {gamma}"


@pytest.mark.parametrize(
    "max_gamma,horizon,update_interval",
    [
        (0.99, 1000, 100),
        (0.95, 500, 50),
        (0.999, 2000, 200),
    ],
)
def test_logarithmic_gamma_constant_between_update_intervals(
    max_gamma: float,
    horizon: int,
    update_interval: int,
):
    """Tests that gamma stays constant between update intervals."""
    scheduler = LogarithmicGammaScheduler(
        max_gamma=max_gamma,
        horizon=horizon,
        update_interval=update_interval,
    )

    # Test that gamma stays constant within each update interval
    max_step = horizon * 2
    prev_gamma = None

    for step in range(max_step):
        gamma = scheduler.get_gamma(step)

        # Gamma should only change at non-boundary steps
        if step % update_interval != 0 and step > 0:
            # Within an interval, gamma should stay constant
            if prev_gamma is not None:
                assert (
                    gamma == prev_gamma
                ), f"Gamma changed within interval at step {step}: {prev_gamma} -> {gamma}"

        prev_gamma = gamma
