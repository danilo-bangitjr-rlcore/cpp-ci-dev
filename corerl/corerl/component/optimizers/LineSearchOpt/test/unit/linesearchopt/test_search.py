import linesearchopt.search as search


# -------------------------
# Armijo
# -------------------------
def test_armijo1():
    """
    The Armijo line search condition passes
    """
    c = 0.1
    armijo = search.Armijo(c, 0.95)

    step_size = 0.1
    loss = 10.0
    directional_derivative = 0.2

    thresh = loss + step_size * c * directional_derivative
    loss_next = 0.5 * thresh  # loss < thresh

    passed, _ = armijo.step(step_size, loss, directional_derivative, loss_next)

    assert passed


def test_armijo2():
    """
    The Armijo line search condition does not pass
    """
    c = 0.1
    armijo = search.Armijo(c, 0.95)

    step_size = 0.1
    loss = 10.0
    directional_derivative = 0.2

    thresh = loss + step_size * c * directional_derivative
    loss_next = 1.5 * thresh  # loss < thresh

    passed, _ = armijo.step(step_size, loss, directional_derivative, loss_next)

    assert not passed


# -------------------------
# Goldstein
# -------------------------
def test_goldstein1():
    """
    The Goldstein line search condition passes
    """
    c = 0.5
    goldstein = search.Goldstein(c, 0.95)

    step_size = 0.1
    loss = 10.0
    directional_derivative = 0.2

    upper = loss + step_size * c * directional_derivative
    lower = loss + step_size * (1 - c) * directional_derivative

    loss_next = lower + 0.05 * (upper - lower)

    passed, _ = goldstein.step(
        step_size, loss, directional_derivative, loss_next,
    )

    assert passed


def test_goldstein2():
    """
    The Goldstein line search condition does not pass
    """
    c = 0.8
    goldstein = search.Goldstein(c, 0.95)

    step_size = 0.1
    loss = 10.0
    directional_derivative = 0.2

    upper = loss + step_size * c * directional_derivative
    lower = loss + step_size * (1 - c) * directional_derivative
    loss_next = lower + 1.5 * (upper - lower)

    passed, _ = goldstein.step(
        step_size, loss, directional_derivative, loss_next,
    )

    assert not passed
