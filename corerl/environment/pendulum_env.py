# This file was adapted from the Pendulum implementation in OpenAI Gym:
# https://github.com/openai/gym/tree/master

# Import modules
from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled



class PendulumEnv(gym.Env):
    """
    PendulumEnv is a modified version of the Pendulum-v0 OpenAI Gym
    environment. In this version, the reward is the cosine of the angle
    between the pendulum and its fixed base. The angle is measured vertically
    so that if the pendulum stays straight up, the angle is 0 radians, and
    if the pendulum points straight down, then the angle is π raidans.
    Therefore, the agent will get reward cos(0) = 1 if the pendulum stays
    straight up and reward of cos(π) = -1 if the pendulum stays straight
    down. The goal is to have the pendulum stay straight up as long as
    possible.

    In this version of the Pendulum environment, state features may either
    be encoded as the cosine and sine of the pendulums angle with respect to
    it fixed base (reference axis vertical above the base) and the angular
    velocity, or as the angle itself and the angular velocity. If θ is the
    angle between the pendulum and the positive y-axis (axis straight up above
    the base) and ω is the angular velocity, then the states may be encoded
    as [cos(θ), sin(θ), ω] or as [θ, ω] depending on the argument trig_features
    to the constructor. The encoding [cos(θ), sin(θ), ω] is a somewhat easier
    problem, since cos(θ) is exactly the reward seen in that state.

    Let θ be the angle of the pendulum with respect to the vertical axis from
    the pendulum's base, ω be the angular velocity, and τ be the torque
    applied to the base. Then:
        1. State features are vectors: [cos(θ), sin(θ), ω] if the
           self.trig_features variable is True, else [θ, ω]
        2. Actions are 1-dimensional vectors that denote the torque applied
           to the pendulum's base: τ ∈ [-2, 2]
        3. Reward is the cosine of the pendulum with respect to the fixed
           base, measured with respect to the vertical axis proceeding above
           the pendulum's base: cos(θ)
        4. The start state is always with the pendulum horizontal, pointing to
           the right, with 0 angular velocity

    Note that this is a continuing task.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0, continuous_action=True, trig_features=False):
        """
        Constructor

        Parameters
        ----------
        g : float, optional
            Gravity, by default 10.0
        trig_features : bool
            Whether to use trigonometric encodings of features or to use the
            angle itself, by default False. If True, then state features are
            [cos(θ), sin(θ), ω], else state features are [θ, ω] (see class
            documentation)
        seed : int
            The seed with which to seed the environment, by default None

        """
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.length = 1.
        self.viewer = None
        self.continuous_action = continuous_action

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        # Set the actions
        if self.continuous_action:
            self.action_space = spaces.Box(
                low=-self.max_torque,
                high=self.max_torque, shape=(1,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(3)

        # Set the states
        self.trig_features = trig_features
        if trig_features:
            # Encode states as [cos(θ), sin(θ), ω]
            high = np.array([1., 1., self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-high,
                high=high,
                dtype=np.float32
            )
        else:
            # Encode states as [θ, ω]
            low = np.array([-np.pi, -self.max_speed], dtype=np.float32)
            high = np.array([np.pi, self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32
            )


    def step(self, u):
        """
        Takes a single environmental step

        Parameters
        ----------
        u : array_like of float
            The torque to apply to the base of the pendulum

        Returns
        -------
        3-tuple of array_like, float, bool, dict
            The state observation, the reward, the done flag (always False),
            and some info about the step
        """
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        length = self.length
        dt = self.dt

        if self.continuous_action:
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
        else:
            u = u[0]
            assert self.action_space.contains(u), \
                f"{u!r} ({type(u)}) invalid"
            u = (u - 1) * self.max_torque  # [-max_torque, 0, max_torque]

        self.last_u = u  # for rendering

        # NOTE: Difference from gymnasium - 3 at the beginning of the below expression is 
        # negated. This is corrected by the addition of pi in the argument of the sine term.
        # This quirk was present in Neumann's GAC repo; keeping it as is.
        newthdot = thdot + (-3 * g / (2 * length) * np.sin(th + np.pi) + 3. /
                            (m * length ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        reward = np.cos(newth)

        if self.trig_features:
            # States are encoded as [cos(θ), sin(θ), ω]
            return self._get_obs(), reward, False, False, {}

        # States are encoded as [θ, ω]
        return self.state, reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to its starting state

        Returns
        -------
        array_like of float
            The starting state feature representation
        """
        super().reset(seed=seed)
        # Note, gymnasium version of pendulum randomly samples the initial state.
        # This version (adapted from GAC paper) uses a fixed initial state.
        state = np.array([np.pi, 0.])
        self.state = angle_normalize(state)
        self.last_u = None

        if self.trig_features:
            # States are encoded as [cos(θ), sin(θ), ω]
            return self._get_obs(), {}

        # States are encoded as [θ, ω]
        return self.state, {}

    def _get_obs(self):
        """
        Creates and returns the state feature vector

        Returns
        -------
        array_like of float
            The state feature vector
        """
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        """
        Fully copied from gymnasium version of pendulum.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(gym.envs.classic_control.pendulum.__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Fully copied from gymnasium version of pendulum.
        """
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    """
    Normalizes the input angle to the range [-π, π]

    Parameters
    ----------
    x : float
        The angle to normalize

    Returns
    -------
    float
        The normalized angle
    """
    # return x % (2 * np.pi)
    return ((x+np.pi) % (2*np.pi)) - np.pi
