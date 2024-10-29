import copy

from typing import Optional, Sequence

import numpy as np
import sympy as sym
from gymnasium import spaces
from scipy import signal
from scipy.special import softmax


# Observation = setpoint
# Action = [kp1, ti1]
class ThreeTankEnvBase(object):
    def __init__(
        self,
        isoffline: bool,
        seed: Optional[int]=None,
        random_sp=(3,),
    ):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.random_sp = random_sp
        # the list of set points for tank 1
        self.setpoint = self.rng.choice(random_sp)

        self.KP_MAX, self.TAU_MAX = 20, 20
        self.KP_MIN, self.TAU_MIN = 0, 0.1

        self.default_Kp = 0.15
        self.default_tau_I = 20
        self.default_KI = 1 / self.default_tau_I

        self.height_T1_record = []  # list of Tank1 level
        self.flowrate_T1_record = []  # list of Tank1 Flowrate
        self.setpoint_T1_record = []  # list of Tank1 setpoints
        self.kp_record = []  # list of Tank1 Kp
        self.ti_record = []  # list of Tank1 Ti
        self.ep_num = 1  # episode number
        self.old_error1 = 0
        self.new_error1 = 0

        # To calculate MSE
        self.error_sum = 0
        self.no_of_error = 0
        self.time_step = 0  # initial time_step

        # To calculate Variance
        self.flowrate_buffer = []
        self.del_pids = []

        # initialize kp1 and ti1 values
        self.kp1 = 1.2
        self.ti1 = 10
        timespan = np.linspace(0, 100, 101)
        omega = 0.3

        self.cached_flowrate = None
        self.cached_height_T1 = None
        self.cached_setpoint_T1 = None

        # SP varying gain
        # self.sinfunction = 10 * np.sin(omega * timespan) + 2
        # SP varying tau
        # self.sinfunction2 = 15 * np.sin(omega * timespan) + 6
        self.sinfunction = 8 * np.sin(omega * timespan) + 2   # SP varying gain
        self.sinfunction2 = 11 * np.sin(omega * timespan) + 6  # SP varying tau
        self.processgain = self.sinfunction[int(self.setpoint)]
        x = sym.Symbol('x')
        self.processtau = self.sinfunction2[int(self.setpoint)]
        type2 = sym.Poly((self.processtau * x + 1))
        type2_c = list(type2.coeffs())
        type2_c = np.array(type2_c, dtype=float)
        sys2 = signal.TransferFunction([self.processgain], type2_c)
        sys2 = sys2.to_ss()
        sys2 = sys2.to_discrete(1)  # pyright: ignore[reportAttributeAccessIssue]
        self.isoffline = isoffline
        if self.isoffline:
            self.A = sys2.A * 0.9
            self.B = sys2.B * 0.9
            self.C = sys2.C * 0.9
        else:
            self.A = sys2.A
            self.B = sys2.B
            self.C = sys2.C

        # water level of tank 1 in cm
        self.height_T1 = np.asarray([[self.setpoint - 1.]])
        self.xprime = np.asarray([[self.setpoint - 1.]])
        self.flowrate_T1 = (self.C - self.A) / self.B
        self.state_normalizer = 1.  # 10.

        # Define this parameter for keeping the action penalty in clipping
        # action setting
        self.extra_action_penalty = 0

    # resets the environment to initial values

    def reinit_the_system(self):
        timespan = np.linspace(0, 100, 101)
        omega = 0.3
        self.sinfunction = 8 * np.sin(omega * timespan) + 2  # 10
        # 15 SP varying tau
        self.sinfunction2 = 11 * np.sin(omega * timespan) + 6

        self.processgain = self.sinfunction[int(self.setpoint)]
        x = sym.Symbol('x')
        self.processtau = self.sinfunction2[int(self.setpoint)]

        # self.processtau = 20
        type2 = sym.Poly((self.processtau * x + 1))
        type2_c = list(type2.coeffs())
        type2_c = np.array(type2_c, dtype=float)
        sys2 = signal.TransferFunction([self.processgain], type2_c)
        sys2 = sys2.to_ss()
        sys2 = sys2.to_discrete(1)  # pyright: ignore[reportAttributeAccessIssue]

        if self.isoffline:
            self.A = sys2.A * 0.9
            self.B = sys2.B * 0.9
            self.C = sys2.C * 0.9
        else:
            self.A = sys2.A
            self.B = sys2.B
            self.C = sys2.C

    def reset_reward(self):
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []

    def reset(self, seed=None):
        # Overwriting old seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # the list of set points for tank 1
        self.setpoint = self.rng.choice(self.random_sp)

        # This method resets the model and define the initial values of each
        # property
        # Values calculated to be stable at 35% flowrate (below first valve)
        # self.height_T1 = np.asarray([[0.]])

        # water level of tank 1 in cm
        self.height_T1 = np.asarray([[self.setpoint - 1.]]) / self.C
        self.xprime = np.asarray([[self.setpoint - 1.]]) / self.C
        self.flowrate_T1 = (self.C - self.A) / self.B

        self.breach = 0

        # initialize PID settings
        self.kp1 = 1.2  # 1.2
        self.ti1 = 10  # 15

        self.time_step = 0  # initial time_step
        self.old_error1 = 0  # initialize errors as zeros
        # normalized error between the water level in tank 1 and the set point
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []
        error_T1 = self.setpoint - self.height_T1
        self.no_of_error += 1  # Increament the number of error stored by 1
        self.error_sum += np.square(error_T1).item()  # Sum of error square
        self.new_error1 = error_T1

        self.height_T1_record = []
        self.flowrate_T1_record = []
        self.setpoint_T1_record = []
        self.kp_record = []
        self.ti_record = []

        current_state = [self.setpoint / self.state_normalizer]  # 100 = max
        return np.asarray(current_state), {}

    def update_pid(self, pi_parameters, KI=False):
        # This method update the pid settings based on the action
        self.kp1 = pi_parameters[0]
        if KI:
            self.ti1 = 1 / pi_parameters[1]
        else:
            self.ti1 = pi_parameters[1]

        self.kp1 = np.clip(self.kp1, self.KP_MIN, self.KP_MAX)
        self.ti1 = np.clip(self.ti1, self.TAU_MIN, self.TAU_MAX)

    def pid_controller(self):
        # This method calculates the PID results based on the errors and PID
        # parameters. Uses velocity form of the equation
        del_fr_1 = self.kp1 * (
            self.new_error1 - self.old_error1 + self.new_error1 /
            (self.ti1 + 1e-8)
        )
        del_flow_rate = [del_fr_1]
        # self.flowrate_1_buffer.append(del_fr_1)
        return np.asarray(del_flow_rate)

    def get_setpoints(self):
        return self.setpoint

    # changes the set points
    def set_setpoints(self, setpoints_T1=None):
        if setpoints_T1 is not None:
            self.setpoint = setpoints_T1

    # the environment reacts to the inputted action
    def inner_step(self, delta_flow_rate, disturbance=0):
        # updating the flow rate of pump 1 given the change in flow rate
        self.flowrate_T1 += delta_flow_rate[0]

        self.height_T1 = self.height_T1

        # bounds the flow rate of pump 1 between 0% and 100%
        self.flowrate_T1 = np.clip(self.flowrate_T1, 0, 100)

        setpoint_T1 = self.setpoint

        self.height_T1 = self.xprime
        self.xprime = self.height_T1 * self.A + self.flowrate_T1 * self.B
        self.height_T1 = self.height_T1 * self.C
        self.height_T1 = np.clip(self.height_T1, 0, 43.1)

        self.height_T1_record.append(self.height_T1.item())
        self.flowrate_T1_record.append(self.flowrate_T1.item())
        self.setpoint_T1_record.append(setpoint_T1)
        self.kp_record.append(self.kp1)  # store the current kp
        self.ti_record.append(self.ti1)  # store the current ti1

        # calculates the difference between the current water level and its set
        # point in tanks 1 and 3 store error as old error since it will be
        # updated soon
        self.old_error1 = self.new_error1
        error_T1 = setpoint_T1 - self.height_T1
        self.no_of_error += 1
        self.error_sum += np.square(error_T1).item()
        self.new_error1 = error_T1
        # normalizes the heights and errors and returns them as the
        # environment's state
        next_state = [self.setpoint / self.state_normalizer]

        self.time_step += 1  # updates elapsed time

        return np.asarray(next_state)

    def get_mse(self):
        # This method calculates all required factors for reward calculation
        # Sum of error square over the number of errors
        mse = self.error_sum / self.no_of_error
        reward = -mse * 100
        return reward


class ThreeTankEnv(ThreeTankEnvBase):
    # Bandit setting
    def __init__(self, seed=None, baseline_scale=1, random_sp=(3,)):
        super(ThreeTankEnv, self).__init__(
            True, seed=seed, random_sp=random_sp,
        )
        self.observation_space = spaces.Discrete(
            len(random_sp), start=int(np.array(random_sp).min()),
        )

        self.prev_pid_params = None

        self.max_actions = np.array([1, 0.8], dtype=np.float32)
        self.min_actions = np.array([0.01, 0.05], dtype=np.float32)
        self.action_dim = len(self.max_actions)
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.internal_iterations = 500

        self.reset()

    def step(self, a, cache=True, use_baseline=True):
        pid = a
        self.update_pid(pid, KI=True)
        self.prev_pid_params = pid

        params_reset = False
        reset_at = -1
        overfill = False
        sp = self.random_sp
        for _ in range(self.internal_iterations):
            sp = self.inner_step(self.pid_controller())

            _ht1 = np.array(self.height_T1_record)
            _sp = np.array(self.setpoint_T1_record)
            overfill = np.any(_ht1 > 1.075 * _sp)

            if overfill and not params_reset:
                self.update_pid([
                    self.default_Kp, self.default_KI
                ], KI=True)
                params_reset = True
                reset_at = self.time_step

        if cache:
            self.cached_flowrate = copy.deepcopy(self.flowrate_T1_record)
            self.cached_height_T1 = copy.deepcopy(self.height_T1_record)
            self.cached_setpoint_T1 = copy.deepcopy(self.setpoint_T1_record)

        # The normalization step from
        # https://github.com/oguzhan-dogru/RL_PID_Tuning/blob/main/main.py
        r = self.get_reward() / 20
        r = (r + 8) / 8
        if params_reset or overfill:
            r -= 100

        setpoint = np.array(self.setpoint_T1_record)
        process = np.array(self.height_T1_record)
        info = {
            "environment_pid": pid,
            "interval_log": self.height_T1_record,
            "percent_overshoot": percent_overshoot(process, setpoint),
            "rise_time": rise_time(process),
            "settling_time": settling_time(process),
            "params_reset": params_reset,
            "reset_at": reset_at,
        }

        return sp, r, True, False, info

    def reset(self, seed=None):
        out = super(ThreeTankEnv, self).reset(seed)
        self.reset_reward()
        self.reinit_the_system()
        return out

    def get_reward(self):
        # This method calculates all required factors for reward calculation
        # reward = self.get_mse()
        # self.error_sum = 0
        # self.no_of_error = 0
        # self.flowrate_buffer = []
        return self.get_mse()


class TTChangeAction(ThreeTankEnvBase):
    # Normal RL setting.
    def __init__(
        self,
        seed=None,
        reset_to_high_reward=True,
        reset_buffer_size=1,
        reset_buffer_add_delay=25,
        reset_buffer_always_include_start=True,
        mse_penalty_reward=True,
        reset_temperature=np.inf,
        n_internal_iter=500,
    ):
        super(TTChangeAction, self).__init__(False, seed, random_sp=[3])

        self.prev_pid_params = np.array([self.default_Kp, self.default_KI])

        self.mse_penalty_reward = mse_penalty_reward

        self.internal_iterations = n_internal_iter

        self.observation_space = spaces.Box(
            low=np.array([self.KP_MIN, 1 / self.TAU_MAX]),
            high=np.array([self.KP_MAX, 1 / self.TAU_MIN]),
            shape=(2,),
            dtype=np.float32,
        )

        self.curr_step = 0

        self.reset_to_high_reward = reset_to_high_reward
        if reset_to_high_reward:
            # Keep a buffer to hold states which we can reset back to
            self.reset_buffer_always_include_start = \
                reset_buffer_always_include_start

            self.reset_temperature = reset_temperature

            self.reset_buffer_size = reset_buffer_size
            size = (reset_buffer_size, self.observation_space.shape[0])

            self.reset_buffer_add_delay = reset_buffer_add_delay
            self.last_reset_time = 0

            self._reset_buffer = np.empty(size)
            self._reset_buffer_priorities = np.empty(reset_buffer_size)
            self._reset_buffer_priorities[:] = -np.inf

            self.tracking_threshold = -np.inf
            self.start_state = np.array([self.default_Kp, self.default_KI])
            self.start_state_r = self._step(self.start_state, False, False)[1]
            self.tracking_threshold = int(self.start_state_r) - 1

            self._reset_buffer[0, :] = self.start_state
            self._reset_buffer_priorities[0] = self.start_state_r
            self.reset_buffer_pointer = 1

        # Make Three Tank Env an OpenAI Gym Env
        action_scale = 0.2
        self.max_actions = np.array(
            [
                action_scale * self.default_Kp,
                action_scale * self.default_KI,
            ],
            dtype=np.float32,
        )
        self.min_actions = -np.array(
            [
                action_scale * self.default_Kp,
                action_scale * self.default_KI,
            ],
            dtype=np.float32,
        )
        self.action_dim = len(self.max_actions)
        self.action_space = spaces.Box(
            low=np.zeros_like(self.min_actions),
            high=np.ones_like(self.max_actions),
            shape=(self.action_dim,),
        )

    def reset_system(self):
        # water level of tank 1 in cm
        self.height_T1 = np.asarray([[self.setpoint - 1.]]) / self.C
        self.xprime = np.asarray([[self.setpoint - 1.]]) / self.C
        self.flowrate_T1 = (self.C - self.A) / self.B

        self.breach = 0

        self.time_step = 0  # initial time_step
        self.old_error1 = 0  # initialize errors as zeros
        # normalized error between the water level in tank 1 and the set point
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []
        error_T1 = self.setpoint - self.height_T1
        self.no_of_error += 1  # Increament the number of error stored by 1
        self.error_sum += np.square(error_T1).item()  # Sum of error square
        self.new_error1 = error_T1

        self.height_T1_record = []
        self.flowrate_T1_record = []
        self.setpoint_T1_record = []
        self.kp_record = []
        self.ti_record = []

    def get_next_pid_params(self, a):
        pid_params = a + self.prev_pid_params
        pid_params = self.pid_param_clip(pid_params)
        return pid_params

    def get_reward(self):
        if self.mse_penalty_reward:
            r = self.get_mse()
            r /= 15
        else:
            process = np.array(self.height_T1_record)
            setpoint = np.array(self.setpoint_T1_record)

            log_overshoot = 1.03
            overshoot_coeff = 1
            overshoot = percent_overshoot(process, setpoint)
            overshoot_power = (100 * overshoot)
            overshoot_penalty = -overshoot_coeff * (
                log_overshoot ** overshoot_power
            )

            settling_coeff = 1
            settling = settling_time(process, setpoint, True, p=0.05)
            settling_bonus = settling_coeff * settling

            rise_coeff = 1
            rise = rise_time(process, setpoint, True)
            rise_penalty = -rise_coeff * rise

            mse_coeff = 1 / 15
            mse = self.get_mse() * mse_coeff

            r = mse + overshoot_penalty + settling_bonus + rise_penalty
        return r

    def add_to_buffer(self, r, pid_params):
        if not self.reset_to_high_reward:
            return

        higher_reward = r > self._reset_buffer_priorities
        if self.reset_buffer_always_include_start:
            higher_reward = higher_reward[1:]
        can_add_to_buffer = (
            self.curr_step - self.last_reset_time > self.reset_buffer_add_delay
        )
        if can_add_to_buffer and np.any(higher_reward):
            # Update the reset buffer
            if self.reset_buffer_pointer < self.reset_buffer_size:
                ind = self.reset_buffer_pointer

                # Pointer never exceeds buffer length and does not wrap around
                # since we replace elements in the buffer based on reward
                # level, not index.
                self.reset_buffer_pointer += 1
            else:
                indices = np.argwhere(higher_reward).ravel()
                if self.reset_buffer_always_include_start:
                    indices += 1
                ind = self.rng.choice(indices)

            self._reset_buffer[ind, :] = pid_params
            self._reset_buffer_priorities[ind] = r
            self.last_reset_time = self.curr_step

    @property
    def _reset_buffer_full(self):
        return self.reset_buffer_pointer >= self._reset_buffer.shape[0]

    @property
    def _reset_prob(self):
        p = self._reset_buffer_priorities[:self.reset_buffer_pointer]
        temperature = self.reset_temperature
        if temperature == 0:
            probs = np.zeros_like(p)
            probs[p.argmax()] = 1
        elif temperature == np.inf:
            probs = np.ones_like(p)
            probs /= probs.shape[0]
        else:
            p /= temperature
            probs = softmax(p)

        return probs

    @property
    def _reset_ind(self):
        p = self._reset_prob
        return np.random.choice(p.shape[0], p=p)

    @property
    def _reset_state(self):
        ind = self._reset_ind
        return self._reset_buffer[ind, :]

    def _step(self, pid_params, cache=True, add=True):
        self.update_pid(pid_params, KI=True)
        for _ in range(self.internal_iterations):
            self.inner_step(self.pid_controller())

        if cache:
            self.cached_flowrate = copy.deepcopy(self.flowrate_T1_record)
            self.cached_height_T1 = copy.deepcopy(self.height_T1_record)
            self.cached_setpoint_T1 = copy.deepcopy(self.setpoint_T1_record)

        r = self.get_reward()

        ht1 = np.array(self.height_T1_record)
        sp = np.array(self.setpoint_T1_record)
        overfill = np.any(ht1 > 1.075 * sp)
        unstable_params = bool(overfill or r < self.tracking_threshold)
        reset = unstable_params or np.any(pid_params < 0)

        if reset:
            # Reset PID parameters
            if self.reset_to_high_reward:
                pid_params = self._reset_state
            else:
                pid_params = np.array([self.default_Kp, self.default_KI])
            r -= 10
        elif add:
            self.add_to_buffer(r, self.prev_pid_params)

        s_next = self.observation(pid_params)
        self.prev_pid_params = pid_params

        setpoint = np.array(self.setpoint_T1_record)
        process = np.array(self.height_T1_record)
        info = {
            "unstable": unstable_params,
            "percent_overshoot": percent_overshoot(process, setpoint),
            "rise_time": rise_time(process),
            "settling_time": settling_time(process),
        }

        self.reset_reward()
        self.reset_system()

        return s_next, r, False, False, info

    def step(self, a, cache=True):
        assert self.action_space.contains(a)
        assert np.all(a <= 1)
        self.curr_step += 1

        pid_change = a * (
            self.max_actions - self.min_actions
        ) + self.min_actions
        pid_params = self.get_next_pid_params(pid_change)

        return self._step(pid_params, cache=cache)

    def reset(self, seed=None):
        s = self.observation(self.prev_pid_params)
        return s, {}

    def pid_param_clip(self, pid_params):
        pid_params[0] = np.clip(pid_params[0], self.KP_MIN, self.KP_MAX)
        pid_params[1] = np.clip(
            pid_params[1], 1 / self.TAU_MAX, 1 / self.TAU_MIN,
        )
        return pid_params

    def observation(self, pid_params):
        return pid_params


def percent_overshoot(process, setpoint):
    overshoot = process - setpoint
    overshoot = overshoot[overshoot > 0]
    if len(overshoot) == 0:
        return 0
    else:
        i = overshoot.argmax()
        sp = setpoint[i] if isinstance(setpoint, Sequence) else setpoint
        return overshoot[i] / sp


def rise_time(process, setpoint=None, use_setpoint=False, p=0.1):
    assert 0 < p < 1
    assert process.ndim == 1

    if use_setpoint:
        final = setpoint
    else:
        final = process[-1]
    assert isinstance(final, float)

    rise_start = p * final
    rise_end = (1 - p) * final

    t0 = (process < rise_start).argmin()
    t1 = (process > rise_end).argmax()

    _rise_time = t1 - t0
    process_time = len(process)

    return _rise_time / process_time


def settling_time(process, setpoint=None, use_setpoint=False, p=0.05):
    assert 0 < p < 1
    assert process.ndim == 1

    if use_setpoint:
        final = setpoint
    else:
        final = process[-1]

    assert isinstance(final, float)

    low = (1 - p) * final
    high = (1 + p) * final

    # Starting from the end of the process and working toward the beginning,
    # find the first occurrence at which the process was not within p% of its
    # final value
    indicator = np.logical_and(low < process, process < high)
    _settling_time = indicator[::-1].argmin()
    process_time = len(process)

    return _settling_time / process_time
