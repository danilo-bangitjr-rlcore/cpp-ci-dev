from collections import deque
from dataclasses import dataclass

import jax


@dataclass
class Step:
    state: jax.Array
    action: jax.Array
    reward: float
    next_state: jax.Array
    done: bool
    gamma: float


@dataclass
class Transition:
    steps: list[Step]
    n_step_reward: float
    n_step_gamma: float

    @property
    def prior(self):
        return self.steps[0]

    @property
    def post(self):
        return self.steps[-1]

    @property
    def n_steps(self) -> int:
        return len(self.steps) - 1

    @property
    def state_dim(self) -> int:
        return self.prior.state.shape[0]

    @property
    def action_dim(self) -> int:
        return self.prior.action.shape[0]


type StepInfo = dict[int, deque[Step]]


def get_n_step_reward(step_q: deque[Step]) -> tuple[float, float]:
    steps = step_q.copy()  # deque is mutable
    steps.popleft()  # drop first step, it does not contribute to return

    partial_return = 0
    discount = 1

    while len(steps) > 0:
        step = steps.popleft()
        partial_return += discount * step.reward
        discount *= step.gamma
        if step.done:
            break

    return partial_return, discount


def _reset_step_info(min_n_step: int, max_n_step: int) -> StepInfo:
    return {
        n: deque(maxlen=n+1) for n in range(min_n_step, max_n_step + 1)
    }


class TransitionCreator:
    def __init__(self, min_n_step: int = 1, max_n_step: int = 1, gamma: float = 1.0):
        self.min_n_step = min_n_step
        self.max_n_step = max_n_step
        self.gamma = gamma
        assert self.min_n_step > 0
        assert self.max_n_step >= self.min_n_step

        self.step_info = _reset_step_info(self.min_n_step, self.max_n_step)

    def __call__(self,
                 state: jax.Array,
                 action: jax.Array,
                 reward: float,
                 next_state: jax.Array,
                 done: bool) -> list[Transition]:
        step = Step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            gamma=self.gamma
        )

        transitions = []
        for n in range(self.min_n_step, self.max_n_step + 1):
            step_q = self.step_info[n]
            step_q.append(step)

            if len(step_q) >= self.min_n_step + 1:  # +1 because we need start and end states
                if len(step_q) == step_q.maxlen or done:
                    n_step_reward, n_step_gamma = get_n_step_reward(step_q)
                    new_transition = Transition(
                        steps=list(step_q),
                        n_step_reward=n_step_reward,
                        n_step_gamma=n_step_gamma
                    )
                    transitions.append(new_transition)

        if done:
            self.step_info = _reset_step_info(self.min_n_step, self.max_n_step)

        return transitions
