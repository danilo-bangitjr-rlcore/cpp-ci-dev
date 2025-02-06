import pytest
import linesearchopt.init as init


@pytest.fixture
def inputs():
    return tuple(10 ** -exp for exp in range(0, 5))


# -----------------------------------------------------
# ----- StepsizeInit works to initialize searches -----
# -----------------------------------------------------

def test_identity(inputs):
    initializer = init.Identity()

    for input_ in inputs:
        got = initializer.reinit(input_)
        expected = input_
        print(got, expected)
        assert got == expected


def test_to(inputs):
    for to in (0., 1., -1.):
        initializer = init.To(to)

        for input_ in inputs:
            got = initializer.reinit(input_)
            expected = to
            assert got == expected


def test_multiply(inputs):
    for factor in (1.001, 1.01, 1.1, 2., 10.):
        initializer = init.Multiply(factor)

        for input_ in inputs:
            got = initializer.reinit(input_)
            expected = factor * input_
            assert got == expected


def test_power(inputs):
    for degree in (-0.001, -0.01, -0.1, 0.0, 0.001, 0.01, 0.1):
        initializer = init.Power(degree)

        for input_ in inputs:
            got = initializer.reinit(input_)
            expected = input_ ** degree
            assert got == expected


def test_max_previous(inputs):
    init_step_size = 1e-10
    initializer = init.MaxPrevious(init_step_size)

    last = init_step_size
    for input_ in inputs:
        # Record that the stepsize was used before re-initializing the stepsize
        initializer.record_used(input_)

        got = initializer.reinit(input_)
        expected = max(last, input_)
        last = got

        assert got == expected


def test_if_else_func(inputs):
    """
    IfElse works with functions
    """
    thresh = inputs[len(inputs) // 2]
    def condition(x: float) -> bool:
        return x > thresh

    def if_path(_: float) -> float:
        return 0.1

    def else_path(_: float) -> float:
        return -0.1

    initializer = init.IfElse(condition, if_path, else_path)

    for input_ in inputs:
        # Record that the stepsize was used before re-initializing the stepsize
        initializer.record_used(input_)

        got = initializer.reinit(input_)

        if condition(input_):
            expected = if_path(input_)
        else:
            expected = else_path(input_)

        assert got == expected


def test_if_else_init(inputs):
    """
    IfElse works with init.StepsizeInit
    """
    thresh = inputs[len(inputs) // 2]
    def condition(x: float) -> bool:
        return x > thresh

    if_path = init.To(0.1)
    else_path = init.To(-0.1)
    initializer = init.IfElse(condition, if_path, else_path)

    for input_ in inputs:
        # Record that the stepsize was used before re-initializing the stepsize
        initializer.record_used(input_)

        got = initializer.reinit(input_)

        if condition(input_):
            expected = if_path(input_)
        else:
            expected = else_path(input_)

        assert got == expected


def test_simple_queue(inputs):
    init_queue = [0.1, 0.2]
    initializer = init.SimpleQueue(init_queue)
    queue = [*init_queue, *inputs]

    for i, input_ in enumerate(inputs):
        # Record that the stepsize was used before re-initializing the stepsize
        initializer.record_used(input_)

        got = initializer.reinit(input_)
        expected = queue[i]

        assert got == expected

    # The queue should always have len(init_queue) size
    assert initializer._queue.qsize() == len(init_queue)


def test_priority_queue(inputs):
    init_queue = [0.1, 0.2]
    initializer = init.PriorityQueue(init_queue)
    queue = [*init_queue]

    for input_ in inputs:
        # Record that the stepsize was used before re-initializing the stepsize
        initializer.record_used(input_)

        queue.append(input_)

        got = initializer.reinit(input_)
        expected = max(queue)
        queue.remove(expected)

        assert got == expected

    # The queue should always have len(init_queue) size
    assert initializer._queue.qsize() == len(init_queue)
