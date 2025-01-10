from typing import Callable

import numpy as np

from corerl.data_pipeline.data_utils.exp_moving import ExpMovingAvg, ExpMovingVar


#######################################
# Exponential Moving Average
#######################################
def test_return_constant_val():
    x = 1.0
    alpha = 0.9
    ema = ExpMovingAvg(alpha)
    mu = None
    for _ in range(3):
        ema.feed(np.array([x]))

    mu = ema()
    assert np.isclose(1.0, mu)


def test_9_to_1_result():
    """
    feeding 9 -> 8 -> ... -> 2 -> 1 with alpha = 0.1 should result in
    mu ~= (1 - 0.1) * [1 + 0.1 * 2 + 0.1^2 * 3 + ... + 0.1^8 * 9]
    = 0.9 * [1 + 0.2 + 0.03 + ...]
    """
    alpha = 0.1
    ema = ExpMovingAvg(alpha)
    mu = None
    for x in range(9, 0, -1):
        ema.feed(np.array([x]))

    mu = ema()
    assert np.isclose(mu, 0.9 * 1.23456789)


def get_spread_batch(mu: float, n: int) -> np.ndarray:
    """
    returns a batch spread evenly around mu
    """
    left = [mu - (i + 1) for i in range(n)]
    right = [mu + (i + 1) for i in range(n)]
    batch = left + [mu] + right
    batch = np.array(batch)
    assert np.isclose(mu, batch.mean())

    return batch


def succeed_multiple_times(test: Callable[[], None], n: int, max_attempts: int):
    """
    succeed n times in a row to pass
    """
    successes = 0
    attempts = 0
    while attempts < max_attempts:
        try:
            test()
            successes += 1
        except Exception:
            successes = 0

        if successes >= n:
            return

        attempts += 1

    raise AssertionError


#######################################
# Exponential Moving Variance
#######################################


def test_longrun_exp_moving_var():
    succeed_multiple_times(try_test_longrun_exp_moving_var, n=2, max_attempts=10)


def try_test_longrun_exp_moving_var():
    epsilon = 0.05
    alpha = 0.999
    emv = ExpMovingVar(alpha)
    for _ in range(10_000):
        z = np.random.normal()
        emv.feed(np.array([z]))

    # test mean
    assert abs(emv._ema()) < epsilon

    # test var
    var = emv()
    assert abs(var - 1.0) < epsilon


def test_longrun_exp_moving_batch_var():
    succeed_multiple_times(try_longrun_exp_moving_batch_var, n=2, max_attempts=10)


def try_longrun_exp_moving_batch_var():
    epsilon = 0.05
    alpha = 0.999
    emv = ExpMovingVar(alpha)
    for _ in range(10_000):
        z_batch = np.array([np.random.normal() for _ in range(5)])
        emv.feed(z_batch)

    # test mean
    assert abs(emv._ema()) < epsilon

    # test var
    var = emv()
    assert abs(var - 1.0) < epsilon


def test_var_differential():
    alpha = 0.999
    low_var_emv = ExpMovingVar(alpha)
    high_var_emv = ExpMovingVar(alpha)
    for _ in range(10_000):
        z = np.random.normal()
        low_var_emv.feed(np.array([z]))

        x = 2 * np.random.normal()
        high_var_emv.feed(np.array([x]))

    assert low_var_emv() < high_var_emv()


def test_batch_var_differential():
    alpha = 0.999
    low_var_emv = ExpMovingVar(alpha)
    high_var_emv = ExpMovingVar(alpha)
    for _ in range(10_000):
        z_batch = np.array([np.random.normal() for _ in range(5)])
        low_var_emv.feed(z_batch)

        x_batch = np.array([2 * np.random.normal() for _ in range(5)])
        high_var_emv.feed(x_batch)

    assert low_var_emv() < high_var_emv()


def test_var_adaptation():
    succeed_multiple_times(try_var_adaptation, n=2, max_attempts=15)


def try_var_adaptation():
    epsilon = 0.05
    alpha = 0.999
    emv = ExpMovingVar(alpha)

    # start with higher variance
    for _ in range(10_000):
        x = 2 * np.random.normal()
        emv.feed(np.array([x]))

    # test var
    var = emv()
    assert abs(var - 4.0) < 4 * epsilon  # more wiggle room for high var

    # finish with low variance
    for _ in range(10_000):
        z = np.random.normal()
        emv.feed(np.array([z]))

    # test var
    var = emv()
    assert abs(var - 1.0) < epsilon


def test_ema_with_nan_gaps():
    alpha = 0.5
    ema = ExpMovingAvg(alpha)
    ema.feed(np.array([1.0]))
    assert np.isclose(ema(), 1.0)

    # new = (1 - 0.5^1) * 2 + 0.5^1 * 1 = 1.5
    ema.feed(np.array([2.0]))
    assert np.isclose(ema(), 1.5)

    # with nan gaps, the effective alpha is 0.5^3
    # new = (1 - 0.5^3) * 4 + 0.5^3 * 1.5 = 3.6875
    ema.feed(np.array([np.nan, np.nan, 4.0]))
    assert np.isclose(ema(), 3.6875)


def test_emv_with_nan_gaps():
    """
    check that NaN gaps increase the effective alpha decay for variance
    """
    alpha = 0.5
    emv = ExpMovingVar(alpha)

    emv.feed(np.array([1.0]))

    # delta = 2 - initial_var
    # new = (1 - 0.5^1) * delta^2 + 0.5^1 * initial_var
    emv.feed(np.array([2.0]))
    one_step = emv()

    # After 2 NaN gap
    # delta = 4 - one_step
    # new = (1 - 0.5^3) * delta^2 + 0.5^3 * one_step
    emv.feed(np.array([np.nan, np.nan, 4.0]))
    three_step = emv()

    assert three_step > one_step


def test_batch_with_multiple_nan_gaps():
    alpha = 0.5
    ema = ExpMovingAvg(alpha)
    emv = ExpMovingVar(alpha)
    ema.feed(np.array([1.0]))
    emv.feed(np.array([1.0]))

    batch = np.array([2.0, np.nan, np.nan, 4.0, np.nan, 8.0])
    ema.feed(batch)
    emv.feed(batch)

    assert ema() > 4.0  # pulled up by the 8.0
    assert emv() > 0  # should have significant variance due to spread
