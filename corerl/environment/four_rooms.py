# Import modules
from typing import Optional

import matplotlib.pyplot as plt

from corerl.utils.hook import Hooks, when
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class FourRoomsEnv(gym.Env):
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

    def __init__(
        self, seed, continuous_action=True, action_scale=0.01,
        noise_scale=0.0, misleading_reward=False,
        decay_scale=0.25, decay_probability=1.0,
    ):
        self._fig = None
        self._ax = None

        self._misleading_reward = misleading_reward

        self._positive_action_decay = decay_scale
        self._positive_action_decay_prob = decay_probability

        if continuous_action:
            max_action = np.array([1, 1])
            self.action_space = spaces.Box(
                -max_action, max_action, dtype=np.float32,
            )
            self._continuous_action = True
        else:
            self.action_space = spaces.Discrete(5)
            self._continuous_action = False

        self._action_scale = action_scale

        # Scale of 0-mean Gaussian noise to add to each action
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            np.array([0, 0]),
            np.array([1, 1]),
            dtype=np.float32,
        )

        self._current_state = np.array([0.05, 0.05])

        self._hooks = Hooks(keys=[e.value for e in when.Env])

        self._hooks(when.Env.AfterCreate, self)

    def register_hook(self, hook, when: when.Env):
        self._hooks.register(hook, when)

    @classmethod
    def _discrete_to_continuous(cls, action):
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

    def step(self, action):
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

        args, _ = self._hooks(
            when.Env.BeforeStep, self, self._current_state, action,
        )

        prev_state = self._current_state
        prev_x, prev_y = prev_state
        next_state = prev_state + self._action_scale * action
        next_state = np.clip(next_state, 0, 1)
        next_x, next_y = next_state

        def line(t):
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

        if self._misleading_reward:
            reward = -0.001 * np.linalg.norm(self._current_state)
        else:
            reward = 0

        args, _ = self._hooks(
            when.Env.AfterStep,
            self, self._current_state, action, reward, prev_state, False,
            False,
        )

        return self._current_state, reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        args, _ = self._hooks(when.Env.BeforeReset, self, self._current_state)
        self._fig = None
        self._ax = None
        self._current_state = np.array([0.05, 0.05])

        args, _ = self._hooks(when.Env.AfterReset, self, self._current_state)
        self.state = args[1]

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
