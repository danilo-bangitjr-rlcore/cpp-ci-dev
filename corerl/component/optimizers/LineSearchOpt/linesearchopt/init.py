import queue
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Callable, Union

from typing_extensions import override


class StepsizeInit(ABC):
    """Determines how stepsizes are re-initialized between line searches.
    """
    def __call__(self, step_size: float) -> float:
        return self.reinit(step_size)

    def record_used(self, step_size: float) -> None:
        """
        Record that `step_size` was successfully used in an optimisation step.
        """
        return None

    @abstractmethod
    def reinit(self, step_size: float) -> float:
        """Get the stepsize to use at the next line search iteration.
        """
        pass


class Identity(StepsizeInit):
    """Does not re-initialize stepsizes.
    """
    def __init__(self):
        super().__init__()

    @override
    def reinit(self, step_size: float) -> float:
        return step_size


class To(StepsizeInit):
    """Re-initializes stepsizes **to** a fixed value.
    """
    def __init__(self, step_size: float):
        """Initializes the instance based on `step_size`

        Args:
            step_size: the stepsize to re-initializes line searches from
        """
        super().__init__()
        self._step_size = step_size

    @override
    def reinit(self, step_size: float) -> float:
        return self._step_size


class Multiply(StepsizeInit):
    """Re-initializes stepsizes to a multiple of the last used stepsize.

    Let `α` be the stepsize used in the previous line search. Multiply
    re-initializes this stepsize to `kα`, where `k > 1`.
    """
    def __init__(self, factor: float):
        """Initializes the instance based on `factor`

        Args:
            factor: the factor by which to multiply the previous step size.
        """
        super().__init__()
        assert factor > 1
        self._factor = factor

    @override
    def reinit(self, step_size: float) -> float:
        return step_size * self._factor


class Power(StepsizeInit):
    """Re-initializes stepsizes to a power of the last used stepsize.

    Let `α` be the stepsize used in the previous line search. Multiply
    re-initializes this stepsize to `α^ξ`, where `-1 < ξ < 1`.
    """
    def __init__(self, degree: float):
        """Initializes the instance based on `degree`

        Args:
            degree: the degree to raise the previous step size.
        """
        super().__init__()
        assert -1 < degree < 1
        self._degree = degree

    @override
    def reinit(self, step_size: float) -> float:
        return step_size ** self._degree


class MaxPrevious(StepsizeInit):
    """Re-initializes the stepsize to last largest previous stepsize.

    MaxPrevious re-initializes stepsizes the maximum previously recorded
    stepsize.
    """
    def __init__(self, init_step_size: float):
        """Initializes the instance based on `init_step_size`

        Args:
            init_step_size: the initial step size.
        """
        super().__init__()
        self._step_size = init_step_size

    @override
    def record_used(self, step_size: float) -> None:
        self._step_size = max(step_size, self._step_size)

    @override
    def reinit(self, step_size: float) -> float:
        return self._step_size


class IfElse(StepsizeInit):
    """Re-initializes stepsizes based on an if-else condition.

    IfElse re-initializes stepsizes to a function of the last used stepsize
    if the last used stepsize satisfies some condition.

    Let `c: ℝ → {0,1}` and `fᵢ: ℝ → ℝ` for `i = 0, 1` be functions of stepsize
    specified by the user. Let `ς` be the stepsize used for the last gradient
    update. Then, the stepsize is initialized to `fᵢ(ς)` for `i = c(ς)`.

    Parameters
    ----------
    condition : function
        The function `c` above
    if_block : function or StepSizeInit
        The function `f₁` above, which is used to initialize the stepsize if
        `c(ς) == 1`
    else_black : function or StepSizeInit
        The function `f₀` above, which is used to initialize the stepsize if
        `c(ς) == 0`
    """
    def __init__(
        self,
        condition: Callable[[float], bool],
        if_block: Union[StepsizeInit, Callable[[float], float]],
        else_block: Union[StepsizeInit, Callable[[float], float]],
    ):
        """Initializes the instance based on arguments.

        Args:
            condition: a boolean condition on the previous stepsize
            if_block: the way to re-initialize stepsizes if `condition` is
                `True`
            else_block: the way to re-initialize stepsizes if `condition` is
                `False`
        """
        super().__init__()
        self._condition = condition
        self._if_block = if_block
        self._else_block = else_block

    @override
    def reinit(self, step_size: float) -> float:
        if self._condition(step_size):
            return self._if_block(step_size)
        else:
            return self._else_block(step_size)

    @override
    def record_used(self, step_size: float) -> None:
        if isinstance(self._if_block, StepsizeInit):
            self._if_block.record_used(step_size)
        if isinstance(self._else_block, StepsizeInit):
            self._else_block.record_used(step_size)


class SimpleQueue(StepsizeInit):
    """Re-initializes stepsizes from a queue of stepsizes.

    SimpleQueue re-initializes stepsizes using a queue of previously recorded
    stepsizes.

    The queue is initially filled with the step sizes in the `init_step_sizes`
    list. The length of `init_step_sizes` determines the total number of
    stepsizes in the queue at any time when using the default `Optimiser`
    class. If `len(init_step_sizes) == 1`, then `SimpleQueue` behaves similarly
    to `Null`.
    """
    def __init__(self, init_step_sizes: Collection[float]):
        """Initializes the instance based on arguments.

        Args:
            init_step_sizes: the stepsizes to initialize the queue with.
        """
        assert len(init_step_sizes) > 0
        super().__init__()
        self._queue = queue.SimpleQueue()

        for init_step_size in init_step_sizes:
            self._queue.put(init_step_size)

    @override
    def record_used(self, step_size: float) -> None:
        self._queue.put(step_size)

    @override
    def reinit(self, step_size: float) -> float:
        assert not self._queue.empty()
        return self._queue.get()



class PriorityQueue(StepsizeInit):
    """Re-initializes stepsizes from a priority queue of stepsizes.

    PriorityQueue re-initializes stepsizes using a priority queue of previously
    recorded stepsizes.

    The queue is initially filled with the step sizes in the `init_step_sizes`
    list. The length of `init_step_sizes` determines the total number of
    stepsizes in the queue at any time when using the default `Optimiser`
    class.

    The priority queue can either return the maximum of previously recorded
    stepsizes (`max_=True`) or the minimum of previously recorded stepsizes
    (`max_=False`).
    """
    def __init__(self, init_step_sizes: Collection[float], max_: bool=True):
        """Initializes the instance based on arguments.

        Args:
            init_step_sizes: the stepsizes to initialize the queue with.
        """
        assert len(init_step_sizes) > 0
        super().__init__()
        self._queue = queue.PriorityQueue()

        self._max = max_
        for init_step_size in init_step_sizes:
            if max_:
                self._queue.put(-init_step_size)
            else:
                self._queue.put(init_step_size)

    @override
    def record_used(self, step_size: float) -> None:
        if self._max:
            self._queue.put(-step_size)
        else:
            self._queue.put(step_size)

    @override
    def reinit(self, step_size: float) -> float:
        assert not self._queue.empty()
        if self._max:
            return -self._queue.get()
        else:
            return self._queue.get()
