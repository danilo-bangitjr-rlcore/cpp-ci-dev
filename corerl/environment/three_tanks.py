from typing import Any
import numpy as np
import sympy as sym
from scipy import signal
from gymnasium import spaces

# Observation = setpoint
# Action = [kp1, ti1]
class ThreeTankEnvBase(object):
    def __init__(self, isoffline: bool, seed: int | None = None, random_sp: list[int] | None = None):
        if random_sp is None:
            random_sp = [3]

        self.W1 = 0.025
        self.W2 = 0.025
        self.W3 = 1000#6000
        self.W4 = 1000#6000
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.random_sp = random_sp
        self.setpoint = self.rng.choice(random_sp)  # the list of set points for tank 1

        self.Lambda = 0
        self.C1 = 0  # Kp penalty # bug
        self.C2 = 0  # taui penalty # bug
        self.C3 = 0  # CV penalty # bug
        self.C4 = 0  # MV penalty # bug
        self.breach = 0
        self.constrain_contribution = 0  # constrain to be multiplied by lambda

        # Make Three Tank Env an OpenAI Gym Env
        self.max_actions = np.array([20, 20], dtype=np.float32)
        self.min_actions = np.array([0, 0], dtype=np.float32)
        self.action_dim = len(self.max_actions)
        self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))

        self.KP_MAX, self.TAU_MAX, self.MV_MAX, self.CV_MAX = 20, 20, 0.6, self.setpoint * 1.1
        self.KP_MIN, self.TAU_MIN, self.MV_MIN, self.CV_MIN = 0, 0, 0, 0

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
        # self.sinfunction = 10 * np.sin(omega * timespan) + 2   # SP varying gain
        # self.sinfunction2 = 15 * np.sin(omega * timespan) + 6  # SP varying tau
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

        self.height_T1 = np.asarray([[self.setpoint - 1.]])  # water level of tank 1 in cm
        self.xprime = np.asarray([[self.setpoint - 1.]])
        self.flowrate_T1 = (self.C - self.A) / self.B
        self.state_normalizer = 1. #10.

        # Define this parameter for keeping the action penalty in clipping action setting
        self.extra_action_penalty = 0

    # resets the environment to initial values

    def reinit_the_system(self):
        timespan = np.linspace(0, 100, 101)
        omega = 0.3
        self.sinfunction = 8 * np.sin(omega * timespan) + 2  # 10
        self.sinfunction2 = 11 * np.sin(omega * timespan) + 6  # 15 SP varying tau

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
        if seed is not None: # Overwriting old seed
            self.rng = np.random.RandomState(seed)

        self.setpoint = self.rng.choice(self.random_sp)  # the list of set points for tank 1

        # This method resets the model and define the initial values of each property
        # self.height_T1 = np.asarray([[0.]])  # Values calculated to be stable at 35% flowrate (below first valve)
        self.height_T1 = np.asarray([[self.setpoint - 1.]]) / self.C  # water level of tank 1 in cm
        self.xprime = np.asarray([[self.setpoint - 1.]]) / self.C
        self.flowrate_T1 = (self.C - self.A) / self.B

        self.Lambda = 0
        self.C1 = 0  # Kp penalty
        self.C2 = 0  # taui penalty
        self.C3 = 0  # CV penalty
        self.C4 = 0  # MV penalty
        self.breach = 0
        self.constrain_contribution = 0  # constrain to be multiplied by lambda

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

        current_state = [self.setpoint / self.state_normalizer]  # 100. is the max level
        return np.asarray(current_state)

    def update_pid(self, pi_parameters):
        # This method update the pid settings based on the action
        self.kp1 = pi_parameters[0]
        self.ti1 = pi_parameters[1]
        self.kp1 = np.clip(self.kp1, self.KP_MIN, self.KP_MAX)
        self.ti1 = np.clip(self.ti1, self.TAU_MIN, self.TAU_MAX)

    def pid_controller(self):
        # This method calculates the PID results based on the errors and PID parameters.
        # Uses velocity form of the euqation
        del_fr_1 = self.kp1 * (self.new_error1 - self.old_error1 + self.new_error1 / (self.ti1 + 1e-8))
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
        # if no value for the valves is given, the valves default to this configuration
        pump_bound = 0
        self.flowrate_T1 += delta_flow_rate[0]  # updating the flow rate of pump 1 given the change in flow rate

        if self.flowrate_T1 > 100:
            pump_bound += abs(self.flowrate_T1 - 100)
        elif self.flowrate_T1 < 0:
            pump_bound += abs(self.flowrate_T1)

        if disturbance != 5:
            self.height_T1 = self.height_T1

        self.flowrate_T1 = np.clip(self.flowrate_T1, 0, 100)  # bounds the flow rate of pump 1 between 0% and 100%

        setpoint_T1 = self.setpoint

        self.height_T1 = self.xprime
        self.xprime = self.height_T1 * self.A + self.flowrate_T1 * self.B
        self.height_T1 = self.height_T1 * self.C
        self.height_T1 = np.clip(self.height_T1, 0, 43.1)

        if disturbance == 1:
            self.height_T1 = self.height_T1 + 0.1
        elif disturbance == 2:
            self.height_T1 = self.height_T1 + 0.3
        elif disturbance == 3:
            self.height_T1 = self.height_T1 + 0.5
        elif disturbance == 4:
            self.height_T1 = self.height_T1 + 1
        else:
            self.height_T1 = self.height_T1

        if self.kp1 > self.KP_MAX:
            self.C1 = abs(self.kp1 - self.KP_MAX)
        elif self.kp1 < self.KP_MIN:
            self.C1 = abs(self.kp1 - self.KP_MIN)
        if self.ti1 > self.TAU_MAX:
            self.C2 = abs(self.ti1 - self.TAU_MAX)
        elif self.ti1 < self.TAU_MIN:
            self.C2 = abs(self.ti1 - self.TAU_MIN)

        if self.height_T1 > self.CV_MAX:  # MV_MAX
            self.C3 = abs(self.height_T1 - self.CV_MAX)
        elif self.height_T1 < self.CV_MIN:
            self.C3 = abs(self.height_T1 - self.CV_MIN)

        if self.flowrate_T1 > self.MV_MAX:  # MV_MAX
            self.C4 = abs(self.flowrate_T1 - self.MV_MAX)
        elif self.flowrate_T1 < self.MV_MIN:
            self.C4 = abs(self.flowrate_T1 - self.MV_MIN)
        self.constrain_contribution = np.abs(
            self.W1 * self.C1 + self.W2 * self.C2 + self.W3 * self.C3 + self.W4 * self.C4
        )
        self.constrain_info = {
            "C1": self.C1,
            "C2": self.C2,
            "C3": np.asarray(self.C3).squeeze(),
            "C4": np.asarray(self.C4).squeeze(),
            "kp1": self.kp1,
            "tau": self.ti1,
            "height": self.height_T1.squeeze(),
            "flowrate": self.flowrate_T1.squeeze(),
        }

        self.height_T1_record.append(self.height_T1.item())
        self.flowrate_T1_record.append(self.flowrate_T1.item())
        self.setpoint_T1_record.append(setpoint_T1)
        self.kp_record.append(self.kp1)  # store the current kp
        self.ti_record.append(self.ti1)  # store the current ti1

        # calculates the difference between the current water level and its set point in tanks 1 and 3
        # store error as old error since it will be updated soon
        self.old_error1 = self.new_error1
        error_T1 = setpoint_T1 - self.height_T1
        self.no_of_error += 1
        self.error_sum += np.square(error_T1)
        self.new_error1 = error_T1
        # normalizes the heights and errors and returns them as the environment's state
        next_state = [self.setpoint / self.state_normalizer]

        self.time_step += 1  # updates elapsed time
        if self.time_step >= 1000:  # terminates the process if the time elapsed reaches the maximum
            done = True
            self.ep_num += 1
        else:
            done = False
        # returns the next state, reward, and if the episode has terminated or not
        return np.asarray(next_state), done

    def get_reward(self):
        # This method calculates all required factors for reward calculation
        mse = self.error_sum / self.no_of_error  # Sum of error square over the number of errors
        # var_action = np.var(self.flowrate_1_buffer)  # Variance of change in flowrate
        # next_reward_comp = [mse / MSE_MAX, var_action / VAR_MAX, self.breach[0] / EXPLORE_KP,
        #                     self.breach[1] / EXPLORE_TI]  # Normalized based on the max values
        # reward = -self.W1 * abs(next_reward_comp[0]) - self.W2 * abs(next_reward_comp[1]) \
        #          - self.W3 * abs(next_reward_comp[2]) - self.W4 * abs(next_reward_comp[3])

        # # add self.extra_action_penalty for the clipping action setting
        # reward = - mse.item() * 100 - self.Lambda * (self.constrain_contribution + self.extra_action_penalty)
        # do not use constraint in the reward
        reward = - mse.item() * 100 # type: ignore
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []
        return reward

# Observation: setpoint -> 1
# Action: [kp1, ti1] -> 2
class ThreeTankEnv(ThreeTankEnvBase):
    # Bandit setting
    def __init__(self, seed: int | None = None, lr_constrain: float = 0, random_sp: list[int] | None = None):
        super().__init__(True, seed=seed, random_sp=random_sp)
        self.constrain_alpha = 5
        self.ep_constrain = 0
        self.ep_constrain_info = {}
        self.lr_constrain = lr_constrain
        self.observation_space = spaces.Discrete(
            len(self.random_sp),
            start=int(np.array(self.random_sp).min()),
        )
        self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(2,), dtype=np.float32)
        self.visualization_range = [-1, max(15, np.array(random_sp).max()+1)]

    def sum_constrain_info(self):
        for k,v in self.constrain_info.items():
            if k in self.ep_constrain_info:
                self.ep_constrain_info[k].append(v)
            else:
                self.ep_constrain_info[k] = [v]
        return

    def step(self, a) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        pid = a
        self.update_pid(pid)
        sp = None
        for _ in range(1000):
            sp, _ = self.inner_step(self.pid_controller())
        self.ep_constrain += self.constrain_contribution
        self.sum_constrain_info()
        # The normalization step from https://github.com/oguzhan-dogru/RL_PID_Tuning/blob/main/main.py
        r = self.get_reward() / 20
        r = (r + 8) / 8
        done = True

        ep_c_info = self.ep_constrain_info
        if done:
            Loss_c = (self.ep_constrain - self.constrain_alpha)
            self.ep_constrain = 0
            self.ep_constrain_info = {}
            self.Lambda = max(0., self.Lambda + self.lr_constrain * Loss_c)

        assert sp is not None
        return sp, r, done, False, {
            'environment_pid': pid,
            'lambda': self.Lambda,
            'interval_log': self.height_T1_record,
            'constrain_detail': ep_c_info,
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
            _, _ = self.inner_step(self.pid_controller())
        self.setpoint = setpoint_backup
        self.reset_reward()
        self.reinit_the_system()
        return s

    def get_action_samples(self, n=10):
        max_a = self.action_space.high/2 #/ self.action_multiplier # re-scale to the range of agent action
        min_a = self.action_space.low #/ self.action_multiplier # re-scale to the range of agent action
        xs = np.linspace(min_a[0], max_a[0], n)
        ys = np.linspace(min_a[1], max_a[1], n)
        xaxis, yaxis = np.meshgrid(xs, ys)
        shape = xaxis.shape
        xaxis, yaxis = xaxis.reshape((-1, 1)), yaxis.reshape((-1, 1))
        return np.array(np.concatenate([xaxis, yaxis], axis=1)), shape


# Observation: [delta_kp1, delta_ti1, prev_kp1, prev_ti1] -> 4
# Action: [delta_kp1, delta_ti1] -> continuous version -> 2
class TTChangeAction(ThreeTankEnv):
    # Normal RL setting.
    def __init__(
        self,
        seed: int | None = None,
        lr_constrain: float = 0,
        constant_pid: bool = True,
        agent_action_min: float = -np.inf,
        agent_action_max: float = np.inf,
        random_sp: list[int] | None = None,
    ):
        super().__init__(seed, lr_constrain, random_sp=random_sp)
        self.prev_pid = np.array([1.2, 10])
        self.prev_a = np.zeros(2)
        if constant_pid:
            self.internal_timeout = 1
        else:
            self.internal_timeout = 10
        self.internal_iterations = 1000//self.internal_timeout
        self.internal_count = 0
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*2),
            high=np.array([np.inf]*2),
            shape=(2,),
            dtype=np.float32,
        )

        self.agent_action_min = agent_action_min # without considering environment scaler
        self.agent_action_max = agent_action_max # without considering environment scaler

    def preprocess_action(self, a: np.ndarray):
        pid = a + self.prev_pid
        pid, self.extra_action_penalty = self.pid_clip(pid)
        return pid

    def step(self, a: np.ndarray):
        # a: change of pid
        pid = self.preprocess_action(a)
        self.update_pid(pid)
        for _ in range(self.internal_iterations):
            sp, _ = self.inner_step(self.pid_controller())
        self.ep_constrain += self.constrain_contribution
        self.sum_constrain_info()
        r = self.get_reward() / 20
        r = (r + 8) / 8 # The normalization step from main.py
        # If unsafe, reset pid to a safe value
        # Continual learning
        old_pid = pid
        if r < -1:
            pid = np.array([1.2, 10])
        done = False

        sp = self.observation(a, pid)
        self.prev_a = a
        self.prev_pid = pid
        ep_c_info = self.ep_constrain_info
        self.internal_count += 1
        if self.internal_count >= self.internal_timeout:
            self.internal_count = 0
            Loss_c = (self.ep_constrain - self.constrain_alpha)
            self.ep_constrain = 0
            self.ep_constrain_info = {}
            self.Lambda = max(0., self.Lambda + self.lr_constrain * Loss_c)

            info = {'environment_pid': old_pid,
                    'lambda': self.Lambda,
                    'interval_log': self.height_T1_record,
                    'constrain_detail': ep_c_info}

        else:
            info = {'environment_pid': old_pid,
                    'lambda': self.Lambda,
                    'interval_log': [],
                    'constrain_detail': ep_c_info}
        return sp, r, done, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if self.internal_count == 0:
            _, info = super().reset()

        s = self.observation(self.prev_a, self.prev_pid)
        return s

    def pid_clip(self, pid):
        C1, C2 = 0, 0
        if pid[0] > self.KP_MAX:
            C1 = abs(pid[0] - self.KP_MAX)
        elif pid[0] < self.KP_MIN:
            C1 = abs(pid[0] - self.KP_MIN)
        if pid[1] > self.TAU_MAX:
            C2 = abs(pid[1] - self.TAU_MAX)
        elif pid[1] < self.TAU_MIN:
            C2 = abs(pid[1] - self.TAU_MIN)
        extra_action_constrain = np.float64(abs(self.W1 * C1 + self.W2 * C2))

        pid[0] = np.clip(pid[0], 0.2,10)#self.KP_MIN, self.KP_MAX) # clip at 0.2, 10, follow the direct action setting
        pid[1] = np.clip(pid[1], 0.2, 10)#self.KP_MIN, self.TAU_MAX) # clip at 0.2, 10, follow the direct action setting
        return pid, extra_action_constrain

    def observation(self, prev_a, pid):
        pid, _ = self.pid_clip(pid)
        # obs = np.concatenate([prev_a, pid], axis=0)
        obs = pid
        return obs

    def get_action_samples(self, n=10):
        min_a = self.agent_action_min
        max_a = self.agent_action_max
        xs = np.linspace(min_a, max_a, n)
        ys = np.linspace(min_a, max_a, n)
        xaxis, yaxis = np.meshgrid(xs, ys)
        shape = xaxis.shape
        xaxis, yaxis = xaxis.reshape((-1, 1)), yaxis.reshape((-1, 1))
        return np.array(np.concatenate([xaxis, yaxis], axis=1)), shape

# Observation: [delta_kp1, delta_ti1, prev_kp1, prev_ti1] -> 4
# Action: cross product of [delta_kp1, delta_ti1] -> discrete version (three choices per dimension) -> 1
class TTChangeActionDiscrete(TTChangeAction):
    def __init__(
        self,
        delta_step: float,
        seed: int | None = None,
        lr_constrain: float = 0,
        constant_pid: bool = True,
        reward_stay: bool = False,
        random_sp: list[int] | None = None,
    ):
        super().__init__(
            seed,
            lr_constrain,
            constant_pid,
            random_sp=random_sp,
        )
        self.action_list = [
            np.array([0, 0]),
            np.array([0, -delta_step]),
            np.array([0, delta_step]),

            np.array([-delta_step, 0]),
            np.array([-delta_step, -delta_step]),
            np.array([-delta_step, delta_step]),

            np.array([delta_step, 0]),
            np.array([delta_step, -delta_step]),
            np.array([delta_step, delta_step]),
        ]
        self.prev_a = [1]
        self.action_space = spaces.Discrete(9, start=0)
        self.reward_stay = reward_stay

    def step(self, a: np.ndarray):
        sp, r, done, trunc, info = super().step(a)
        if self.reward_stay and round(r, 5) == 1.:
            if a == 0: # stay
                r += 0.5
        return sp, r, done, trunc, info

    def preprocess_action(self, a: np.ndarray):
        a = a[0]
        # norm_pid = self.action_list[a] + self.prev_pid
        # pid = self.action_multiplier * norm_pid
        pid = self.action_list[a] + self.prev_pid
        pid, self.extra_action_penalty = self.pid_clip(pid)
        return pid #, norm_pid

    def observation(self, prev_a, pid):
        prev_a = prev_a[0]
        pid, _ = self.pid_clip(pid)
        # obs = np.concatenate([self.action_list[prev_a], pid], axis=0)
        obs = pid
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if self.internal_count == 0:
            _, info = super().reset()
            # self.prev_pid = np.zeros(2)
            # self.prev_a = 1
        else:
            info = {}
        s = self.observation(self.prev_a, self.prev_pid)
        return s, info

    def get_action_samples(self, n=None):
        samples = np.arange(9).reshape(-1, 1)
        shape = samples.shape
        return samples, shape



# Observation: [0, 0, prev_kp1, prev_ti1] -> 4
# Action: [kp1, ti1] -> continuous version
class TTAction(TTChangeAction):
    def __init__(
        self,
        seed: int | None = None,
        lr_constrain: float = 0,
        constant_pid: bool = True,
        random_sp: list[int] | None = None,
    ):
        super().__init__(seed, lr_constrain, constant_pid, random_sp=random_sp)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            shape=(2,),
            dtype=np.float32,
        )

    def preprocess_action(self, a):
        pid = a
        return pid

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        s, info = super().reset()
        s[:2] = 0
        return s, info

    def step(self, a):
        sp, r, done, trunc, info = super().step(a)
        sp[:2] = 0
        return sp, r, done, trunc, info

    def get_action_samples(self, n=10):
        max_a = self.action_space.high/2 #/ self.action_multiplier # re-scale to the range of agent action
        min_a = self.action_space.low #/ self.action_multiplier # re-scale to the range of agent action
        xs = np.linspace(min_a[0], max_a[0], n)
        ys = np.linspace(min_a[1], max_a[1], n)
        xaxis, yaxis = np.meshgrid(xs, ys)
        shape = xaxis.shape
        xaxis, yaxis = xaxis.reshape((-1, 1)), yaxis.reshape((-1, 1))
        return np.array(np.concatenate([xaxis, yaxis], axis=1)), shape


class NonContexTT(ThreeTankEnv):
    def __init__(self, seed=None, lr_constrain=0, obs=0.):
        """With non-contextual setting, the setpoint must be fixed"""
        super().__init__(seed, lr_constrain, random_sp=[3])
        obs = [obs]
        self.observation_space = spaces.Box(low=np.array(obs),
                                            high=np.array(obs), shape=(len(obs),), dtype=np.float32)
        self.obs = obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        _, _ = super().reset(seed=seed)
        return np.array(self.obs)

    def step(self, a):
        sp, r, done, trunc, info = super().step(a)
        sp = np.array(self.obs)
        return sp, r, done, trunc, info
