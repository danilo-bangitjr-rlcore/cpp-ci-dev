from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import gymnasium as gym
import numpy as np
from lib_config.config import config

from rl_env.group_util import EnvConfig, env_group


@config(frozen=True)
class NonstationaryWorldConfig(EnvConfig):
    name: Literal['NonstationaryWorld-v0'] = 'NonstationaryWorld-v0'

    upstream_sensor_drift: bool = False
    upstream_sensor_step: bool = False
    upstream_dynamics_drift: bool = False
    upstream_dynamics_step: bool = False
    downstream_sensor_drift: bool = False
    downstream_sensor_step: bool = False
    downstream_dynamics_drift: bool = False
    downstream_dynamics_step: bool = False

    # periods of nonstationarity
    upstream_sensor_period: int = 300
    downstream_sensor_period: int = 500
    upstream_dyn_period: int = 700
    downstream_dyn_period: int = 1100

class NoiseParams(NamedTuple):
    mu: float = 0.0
    sigma: float = 0.05

class WalkState(NamedTuple):
    val: float # current value of walk
    nominal: float # nominal value that walk will be biased toward
    stiffness: float # strength of bias
    std: float # standard deviation of walk

class NWState(NamedTuple):
    # physical process variables
    A: WalkState = WalkState(val=0, nominal=0, stiffness=0.05, std=0.2)
    B: float = 0.0
    C: float = 0.5
    # sensor parameters
    theta_a: NoiseParams = NoiseParams()
    theta_b: NoiseParams = NoiseParams()
    theta_c: NoiseParams = NoiseParams(sigma=0)
    # unobserved sources of nonstationarity
    A_bar: float = 0.0
    K: float = 5

def sigmoid(x: float):
    return 1/(1 + np.exp(-x))

def inv_sigmoid(x: float):
    return -np.log(1/x - 1)

x = np.random.normal()
assert np.isclose(x, inv_sigmoid(sigmoid(x)))

B_decay = 0.85
C_decay = 0.85

class Dynamics(NamedTuple):
    f_A: Callable[[WalkState, float], WalkState] = lambda A, A_bar: WalkState(
        val=A.val + np.random.normal(A.stiffness * (A.nominal - A.val), A.std),
        nominal=A_bar,
        stiffness=A.stiffness,
        std=A.std,
    )  # biased random walk
    f_B: Callable[[float, float, float, float], float] = lambda A, B, u, K: B_decay*B + (1-B_decay)*(A + K*u)
    f_C: Callable[[float, float], float] = lambda B, C: sigmoid(C_decay*inv_sigmoid(C) + (1-C_decay)*B)

    f_a: Callable[[float, NoiseParams], float] = lambda A, theta: A + np.random.normal(theta.mu, theta.sigma)
    f_b: Callable[[float, NoiseParams], float] = lambda B, theta: B + np.random.normal(theta.mu, theta.sigma)
    f_c: Callable[[float, NoiseParams], float] = lambda C, theta: C + np.random.normal(theta.mu, theta.sigma)


class NonstationaryWorld(gym.Env):
    def __init__(self, cfg: NonstationaryWorldConfig):
        """
                  ┌───┐
                  │ u │
                  └─┬─┘
           ┌───┐  ┌─▼─┐  ┌───┐
           │ A ├──► B ├──► C │
           └─┬─┘  └─┬─┘  └─┬─┘
           ┌─▼─┐  ┌─▼─┐  ┌─▼─┐
           │ a │  │ b │  │ c │
           └───┘  └───┘  └───┘
        """
        if cfg is None:
            cfg = NonstationaryWorldConfig()
        self._cfg = cfg

        self._dynamics = Dynamics()
        self._state = NWState()
        self._timestep = 0
        self.observation_space = gym.spaces.Box(-np.inf*np.ones(2), np.inf*np.ones(2), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1*np.ones(1), np.ones(1), dtype=np.float64)

        # bounds for nonstationary parameters
        self.nominal_A_min = -1.5
        self.nominal_A_max = 1.5
        self.sensitivity_min = 4
        self.sensitivity_max = 6
        self.a_mu_min = -3
        self.a_mu_max = 1
        self.b_mu_min = -3
        self.b_mu_max = 1

        # periods of nonstationarity
        self.upstream_sensor_period = cfg.upstream_sensor_period
        self.downstream_sensor_period = cfg.downstream_sensor_period
        self.upstream_dyn_period = cfg.upstream_dyn_period
        self.downstream_dyn_period = cfg.downstream_dyn_period

    def _get_obs(self, state: NWState):
        a = self._dynamics.f_a(state.A.val, state.theta_a)
        b = self._dynamics.f_b(state.B, state.theta_b)
        c = self._dynamics.f_c(state.C, state.theta_c)
        return np.asarray([a, b, c])

    def backward_eval(self, state: NWState, u: float):
        """
        This function updates the `state` given the current state and the value of the control
        variable u.

        It evaluates the variables in reverse causal order, so that the inputs to the dynamics functions
        come from the previous timestep.

        For example, when rolling forward in time, C_t+1
        should be computed from B_t, C_t, etc.

                  ┌───┐
                  │ u │
                  └─┬─┘
           ┌───┐  ┌─▼─┐  ┌───┐
           │ A ├──► B ├──► C │
           └─┬─┘  └─┬─┘  └─┬─┘
           ┌─▼─┐  ┌─▼─┐  ┌─▼─┐
           │ a │  │ b │  │ c │
           └───┘  └───┘  └───┘
        """

        C = self._dynamics.f_C(state.B, state.C)
        B = self._dynamics.f_B(state.A.val, state.B, u, state.K) # here use A from previous time step
        A = self._dynamics.f_A(state.A, state.A_bar)

        return NWState(A, B, C, state.theta_a, state.theta_b, state.theta_c, state.A_bar, state.K)

    def _periodic_step(self, period: float):
        """
        square wave with range {0, 1}, with period T, starting at 0
        """
        t = self._timestep
        T = period
        return -0.5 * (np.sign(np.sin(2*np.pi/T * t)) - 1)

    def _periodic_drift(self, period: float):
        """
        cos wave with range [0, 1], with period T, starting at 1
        """
        t = self._timestep
        T = period
        return 0.5 * (np.cos(2*np.pi/T * t) + 1)

    def _scale(self, val:float, min: float, max: float):
        """
        linearly scale from [0, 1] to [min, max]
        """
        return val * (max - min) + min

    # nonstationary effects on process variables
    def _get_A_bar(self):
        step = self._periodic_step(self.upstream_dyn_period) if self._cfg.upstream_dynamics_step else 1
        drift = self._periodic_drift(self.upstream_dyn_period) if self._cfg.upstream_dynamics_drift else 1
        return self._scale(step*drift, self.nominal_A_min, self.nominal_A_max)

    def _get_K(self):
        step = self._periodic_step(self.downstream_dyn_period) if self._cfg.downstream_dynamics_step else 1
        drift = self._periodic_drift(self.downstream_dyn_period) if self._cfg.downstream_dynamics_drift else 1
        return self._scale(step*drift, self.sensitivity_min, self.sensitivity_max)

    # nonstationary effects on sensors
    def _get_mu_a(self):
        step = self._periodic_step(self.upstream_sensor_period) if self._cfg.upstream_sensor_step else 1
        drift = self._periodic_drift(self.upstream_sensor_period) if self._cfg.upstream_sensor_drift else 1
        return self._scale(step*drift, self.a_mu_min, self.a_mu_max)

    def _get_mu_b(self):
        step = self._periodic_step(self.downstream_sensor_period) if self._cfg.downstream_sensor_step else 1
        drift = self._periodic_drift(self.downstream_sensor_period) if self._cfg.downstream_sensor_drift else 1
        return self._scale(step*drift, self.b_mu_min, self.b_mu_max)

    def _update_unobserved(self):
        state = self._state
        self._state = NWState(
            A=state.A,
            B=state.B,
            C=state.C,
            theta_a=NoiseParams(mu=self._get_mu_a()),
            theta_b=NoiseParams(mu=self._get_mu_b()),
            theta_c=state.theta_c,
            A_bar=self._get_A_bar(),
            K=self._get_K(),
        )

    def step(self, action: np.ndarray):
        assert action.shape == (1,)
        self._update_unobserved()
        self._state = self.backward_eval(self._state, action[0])
        self._timestep += 1

        obs = self._get_obs(self._state)
        reward = -np.abs(obs[1])

        return obs, reward, False, False, {}


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self._state = NWState()
        return self._get_obs(self._state), {}

    def close(self):
        pass

env_group.dispatcher(NonstationaryWorldConfig(), NonstationaryWorld)
