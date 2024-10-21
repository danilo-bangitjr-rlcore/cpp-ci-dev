from typing import Any
import numpy as np
from gymnasium import spaces
import copy
import sympy as sym
from scipy import signal


# Observation = setpoint
# Action = [kp1, ti1]
class ThreeTankEnvBase(object):
    def __init__(
        self,
        isoffline: bool,
        seed: int | None = None,
        random_sp: list[int] | None = None,
    ):
        if random_sp is None:
            random_sp = [3]

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.random_sp = random_sp
        # the list of set points for tank 1
        self.setpoint = self.rng.choice(random_sp)

        self.KP_MAX, self.TAU_MAX = 20, 20
        self.KP_MIN, self.TAU_MIN = 0, 0.1

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
        sys2 = sys2.to_discrete(1) # type: ignore
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
        sys2 = sys2.to_discrete(1) # type: ignore

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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
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
        self.error_sum += np.square(error_T1)  # Sum of error square
        self.new_error1 = error_T1

        self.height_T1_record = []
        self.flowrate_T1_record = []
        self.setpoint_T1_record = []
        self.kp_record = []
        self.ti_record = []

        current_state = [self.setpoint / self.state_normalizer]  # 100 = max
        return np.asarray(current_state)

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
        self.error_sum += np.square(error_T1)
        self.new_error1 = error_T1
        # normalizes the heights and errors and returns them as the
        # environment's state
        next_state = [self.setpoint / self.state_normalizer]

        self.time_step += 1  # updates elapsed time

        self.cached_flowrate = copy.deepcopy(self.flowrate_T1_record)
        self.cached_height_T1 = copy.deepcopy(self.height_T1_record)
        self.cached_setpoint_T1 = copy.deepcopy(self.setpoint_T1_record)

        return np.asarray(next_state)

    def get_reward(self):
        # This method calculates all required factors for reward calculation
        # Sum of error square over the number of errors
        mse = self.error_sum / self.no_of_error
        reward = - mse.item() * 100 # type: ignore
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []
        return reward


# Observation: setpoint -> 1
# Action: [kp1, ti1] -> 2
class ThreeTankEnv(ThreeTankEnvBase):
    # Bandit setting
    def __init__(
        self,
        seed: int | None = None,
        random_sp: list[int] | None = None,
    ):
        super(ThreeTankEnv, self).__init__(
            True, seed=seed, random_sp=random_sp,
        )
        self.observation_space = spaces.Discrete(
            len(self.random_sp),
            start=int(np.array(self.random_sp).min()),
        )
        self.max_actions = np.array([20, 20], dtype=np.float32)
        self.min_actions = np.array([0, 0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            shape=(2,),
            dtype=np.float32,
        )
        self.visualization_range = [-1, max(15, np.array(self.random_sp).max() + 1)]

    def step(self, a):
        pid = a
        self.update_pid(pid)

        sp = None
        for _ in range(1000):
            sp = self.inner_step(self.pid_controller())

        # The normalization step from
        # https://github.com/oguzhan-dogru/RL_PID_Tuning/blob/main/main.py
        r = self.get_reward() / 20
        r = (r + 8) / 8
        done = True

        assert sp is not None
        return sp, r, done, False, {
            'environment_pid': pid,
            'interval_log': self.height_T1_record,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        s = super().reset(seed=seed)
        setpoint_backup = self.setpoint
        self.setpoint = self.setpoint - 1

        for _ in range(100):
            self.inner_step(self.pid_controller())

        self.setpoint = setpoint_backup
        self.reset_reward()
        self.reinit_the_system()
        return s


# Observation: [delta_kp1, delta_ti1, prev_kp1, prev_ti1] -> 4
# Action: [delta_kp1, delta_ti1] -> continuous version -> 2
class TTChangeAction(ThreeTankEnvBase):
    # Normal RL setting.
    def __init__(self, seed: int | None = None):
        super(TTChangeAction, self).__init__(isoffline=False, seed=seed, random_sp=[3])

        self.default_Kp = 0.15
        self.default_tau_I = 20
        self.default_KI = 1 / self.default_tau_I
        self.prev_pid_params = np.array([self.default_Kp, self.default_KI])

        self.internal_iterations = 500

        self.observation_space = spaces.Box(
            low=np.array([self.KP_MIN, 1 / self.TAU_MAX]),
            high=np.array([self.KP_MAX, 1 / self.TAU_MIN]),
            shape=(2,),
            dtype=np.float32,
        )

        # Make Three Tank Env an OpenAI Gym Env
        action_scale = 0.2
        self.max_actions = np.array(
            [
                action_scale * self.default_Kp,
                action_scale * self.default_KI,
            ],
            dtype=np.float32,
        )
        self.min_actions = np.array(
            [
                -action_scale * self.default_Kp,
                -action_scale * self.default_KI,
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
        self.error_sum += np.square(error_T1)  # Sum of error square
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

    def step(self, a):
        pid_change = a * (
            self.max_actions - self.min_actions
        ) + self.min_actions
        pid_params = self.get_next_pid_params(pid_change)
        self.update_pid(pid_params, KI=True)
        for _ in range(self.internal_iterations):
            self.inner_step(self.pid_controller())

        r = self.get_reward()
        r /= 15
        # If unsafe, reset pid_params to a safe value
        # Continual learning
        overshoot = (
            max(self.height_T1_record) > 1.05 * self.setpoint_T1_record[-1]
            or r < -2
        )
        if overshoot:
            pid_params = np.array([self.default_Kp, self.default_KI])
            # pid_params = self.prev_pid_params
            r -= 5
        done = False

        s_next = self.observation(pid_params)
        self.prev_pid_params = pid_params

        info = {
            'height_T1_record': self.height_T1_record,
            'setpoint_T1_record': self.setpoint_T1_record,
        }
        self.reset_reward()
        self.reset_system()
        # super(TTChangeAction, self).reset()
        # self.update_pid(pid_params, KI=True)

        return s_next, r, done, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super(TTChangeAction, self).reset()
        s = self.observation(self.prev_pid_params)
        return s

    def pid_param_clip(self, pid_params):

        pid_params[0] = np.clip(pid_params[0], self.KP_MIN, self.KP_MAX)
        pid_params[1] = np.clip(
            pid_params[1], 1 / self.TAU_MAX, 1 / self.TAU_MIN,
        )
        return pid_params

    def observation(self, pid_params):
        return pid_params
