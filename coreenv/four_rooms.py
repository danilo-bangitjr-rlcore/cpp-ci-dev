# Import modules
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


@dataclass
class FourRoomsConfig:
    continuous_action: bool = True
    action_scale: float = 0.01
    noise_scale: float = 0.0
    decay_scale: float = 0.25
    decay_probability: float = 1.0
    seed: int = 0


class FourRoomsEnv(gym.Env):
    """FourRoomsEnv implements the continuous-state four rooms domain with
    discrete or continuous actions.
    """
    corridors_start = [
        (0.225, 0.5),
        (0.725, 0.5),
        (0.5, 0.225),
        (0.5, 0.725),
    ]
    corridors_end = [
        (0.275, 0.5),
        (0.775, 0.5),
        (0.5, 0.275),
        (0.5, 0.775),
    ]

    _EPSILON = 1e-6

    def __init__(self, cfg: FourRoomsConfig | None = None):
        """Initializes the instance

        Args:
            seed:
                The RNG seed to use
            continuous_action:
                Whether actions are continuous or not, by default `True`.
            action_scale:
                Multiplier which determines how far an action moves the agent
            noise_scale:
                One half of the multiplier on 0-mean Gaussian noise added to
                the action. By default 0. Set to 0 to disable.
            decay_scale:
                When decay_probability > 0, actions in the up/right direction
                are decayed by `1 - decay_scale`. This makes it harder for
                the agent to travel up/right. By default 0.25.
            decay_probability:
                Probability with which to decay actions in the up/right
                direction. By default 1.0. Set to 0 to disable.
        """

        if cfg is None:
            cfg = FourRoomsConfig()

        self._fig = None
        self._ax = None

        assert 0 < cfg.decay_scale <= 1
        self._positive_action_decay = cfg.decay_scale
        self._positive_action_decay_prob = cfg.decay_probability

        if cfg.continuous_action:
            max_action = np.array([1, 1])
            self.action_space = spaces.Box(
                -max_action, max_action, dtype=np.float32,
            )
            self._continuous_action = True
        else:
            self.action_space = spaces.Discrete(5)
            self._continuous_action = False

        self._action_scale = cfg.action_scale

        # Scale of 0-mean Gaussian noise to add to each action
        self._noise_scale = cfg.noise_scale
        self._rng = np.random.default_rng(cfg.seed)

        self.observation_space = spaces.Box(
            np.array([0, 0]),
            np.array([1, 1]),
            dtype=np.float32,
        )

        self._current_state = np.array([0.05, 0.05])

    @classmethod
    def _discrete_to_continuous(cls, action: np.ndarray):
        if action.item() == 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        elif action.item() == 1:
            return np.array([-1.0, 0.0], dtype=np.float32)
        elif action.item() == 2:
            return np.array([0.0, 1.0], dtype=np.float32)
        elif action.item() == 3:
            return np.array([0.0, -1.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)
        if not self._continuous_action:
            action = FourRoomsEnv._discrete_to_continuous(action)

        if self._noise_scale > 0:
            action_noise = self._rng.normal(0, 2, (2,)) * self._noise_scale
            action += action_noise

        if self._positive_action_decay_prob > 0:
            eps = self._rng.uniform()
            if eps < self._positive_action_decay_prob:
                action[np.where(action > 0)] *= self._positive_action_decay

        prev_state = self._current_state
        prev_x, prev_y = prev_state
        next_state = prev_state + self._action_scale * action
        next_state = np.clip(next_state, 0, 1)
        next_x, next_y = next_state

        def line(t: float):
            return prev_state + t * (next_state - prev_state)

        # Take the line segment from prev_state -> next_state:
        #
        #       t -> prev_state + t (next_state - prev_state),  t∈[0, 1]
        #
        # Check where this line segment has x=1/2 and where it has y=1/2.
        # Then, check if these points intersect with one of the boundaries
        # at x=1/2, y=1/2.
        if next_x == prev_x:
            # There is no intersection with x boundaries
            t_x = np.nan
        else:
            t_x = (0.5 - prev_x) / (next_x - prev_x)

        if next_y == prev_y:
            # There is no intersection with y boundaries
            t_y = np.nan
        else:
            t_y = (0.5 - prev_y) / (next_y - prev_y)

        x, y = next_x, next_y
        if 0 <= t_x <= 1 and y >= 0.5:  # Crossed the x = 1/2 boundary at top
            start, end = self.corridors_start[-1], self.corridors_end[-1]
            x, y = line(t_x)
            assert x == 1/2
            if not (start[1] <= y <= end[1]):
                # The agent tried to cross over a boundary, not a door
                next_x = 0.5 + np.sign(prev_x - 0.5) * self._EPSILON

        elif 0 <= t_x <= 1 and y <= 0.5:  # Crossed the x = 1/2 boundary at bottom
            start, end = self.corridors_start[-2], self.corridors_end[-2]
            x, y = line(t_x)
            assert x == 1/2
            if not (start[1] <= y <= end[1]):
                # The agent tried to cross over a boundary, not a door
                next_x = 0.5 + np.sign(prev_x - 0.5) * self._EPSILON

        if 0 <= t_y <= 1 and x >= 0.5:  # Cross the y=1/2 boundary on right
            start, end = self.corridors_start[-3], self.corridors_end[-3]
            x, y = line(t_y)
            assert y == 1/2
            if not (start[0] <= x <= end[0]):
                # The agent tried to cross over a boundary, not a door
                next_y = 0.5 + np.sign(prev_y - 0.5) * self._EPSILON

        elif 0 <= t_y <= 1 and x <= 0.5:  # Cross the y=1/2 boundary on left
            start, end = self.corridors_start[-4], self.corridors_end[-4]
            x, y = line(t_y)
            assert y == 1/2
            if not (start[0] <= x <= end[0]):
                # The agent tried to cross over a boundary, not a door
                next_y = 0.5 + np.sign(prev_y - 0.5) * self._EPSILON

        self._current_state = np.array([next_x, next_y], dtype=np.float32)

        reward = 0

        return self._current_state, reward, False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ):
        self._fig = None
        self._ax = None
        self._current_state = np.array([0.05, 0.05])
        self.state = self._current_state

        return self._current_state, {}

    def render(self):
        if self._fig is None:
            plt.ion()
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot()

            for start, end in zip(
                self.corridors_start, self.corridors_end, strict=True,
            ):
                if start[0] < 0.5:
                    self._ax.plot(
                        [0, start[0]], [start[1], start[1]], color="black",
                        linewidth=2,
                    )
                    self._ax.plot(
                        [end[0], 0.5], [end[1], end[1]], color="black",
                        linewidth=2,
                    )
                elif start[0] > 0.5:
                    self._ax.plot(
                        [0.5, start[0]], [start[1], start[1]], color="black",
                        linewidth=2,
                    )
                    self._ax.plot(
                        [end[0], 1], [end[1], end[1]], color="black",
                        linewidth=2,
                    )

                if start[1] < 0.5:
                    self._ax.plot(
                        [start[0], start[0]], [0, start[1]], color="black",
                        linewidth=2,
                    )
                    self._ax.plot(
                        [end[0], end[0]], [end[1], 0.5], color="black",
                        linewidth=2,
                    )
                elif start[1] > 0.5:
                    self._ax.plot(
                        [start[0], start[0]], [0.5, start[1]], color="black",
                        linewidth=2,
                    )
                    self._ax.plot(
                        [end[0], end[0]], [end[1], 1], color="black",
                        linewidth=2,
                    )

            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)

            self._point = self._ax.scatter(
                [self._current_state[0]], [self._current_state[1]],
                color="mediumslateblue",
            )

        self._point.set_offsets([self._current_state])
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
