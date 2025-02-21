from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from gymnasium import Env, spaces
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import signal
from scipy.special import softmax


@dataclass
class PVSConfig:
    reward_type: str = "combined"  # "mse" or "combined"
    use_constraints: bool = False
    use_reset_buffer: bool = True
    reset_buffer_size: int = 10
    internal_iterations: int = 500
    random_sp: Sequence[float] = (4.0, 5.0)
    state_normalizer: float = 1.
    pid_limits: dict = field(default_factory=lambda: {
        "kp_max": 5,
        "kp_min": 0,
        "ti_max": 20,
        "ti_min": 0.1
    })
    default_pid: dict = field(default_factory=lambda: {
        "kp": 0.1,
        "ti": 0.1
    })
    reward_coeffs: dict = field(default_factory=lambda: {
        "mse_coeff": 1/15,
        "overshoot_coeff": 1.5,
        "settling_coeff": 1.2,
        "rise_coeff": 1
    })
    reset_temperature: float = np.inf
    seed: Optional[int] = 1
    no_reset: bool = True


class BasePVSEnv(Env):
    """
    Control Loop
    1. Measure current level (height_T1)
    2. Calculate error = setpoint - current_level
    3. Apply PID control to calculate flow rate change
    4. Update flow rate
    5. System responds to new flow rate
    6. Repeat
    """
    def __init__(self, cfg: PVSConfig):
        super().__init__()
        self.config = cfg
        self.init_random_setpoint(self.config.seed)
        self.init_system_dynamics()
        self.init_transfer_function()
        self.init_state_variables()
        self.init_records()
        self.pid_controller = PIDController(self.config)

    def init_random_setpoint(self, seed: Optional[int]):
        self.rng = np.random.RandomState(seed)
        self.random_sp = self.config.random_sp
        self.setpoint = self.rng.choice(self.random_sp)

    def init_system_dynamics(self):
        """
        Define system dynamics.

        - processgain_func: Used for process gain variation (8 * sin(0.3t) + 2)
        - processtau_func: Used for process time constant variation (11 * sin(0.3t) + 6)
        """
        timespan = np.linspace(0, 100, 101)
        omega = 0.3
        # A1*sin(ωSP) + B1
        self.processgain_func = 8 * np.sin(omega * timespan) + 2
        # A2*sin(ωSP) + B2
        self.processtau_func = 11 * np.sin(omega * timespan) + 6


    def init_transfer_function(self):
        """
        Initialize the system's transfer function and state-space representation.

        Converts the process gain and time constant into a discrete-time state-space model.
        If offline mode is enabled, reduces the system matrices by 10%.
        """
        # according to the paper:
        # The gain and the time constant of this system are a function of the setpoint,
        # thus varying during the experiments. These variations make the system
        # suitable to test the proposed adaptive method.
        self.processgain = self.processgain_func[int(self.setpoint)]
        s = sym.Symbol('s')
        self.processtau = self.processtau_func[int(self.setpoint)]

        # polynomial representation of transfer function denominator: (tau*s + 1)
        transfer_func_denom = sym.Poly((self.processtau * s + 1))
        # extract coefficients from polynomial
        denom_coeffs = list(transfer_func_denom.coeffs())
        denom_coeffs = np.array(denom_coeffs, dtype=float)

        # transfer function: G(s) = process_gain / (tau*s + 1)
        transfer_func: Any = signal.TransferFunction([self.processgain], denom_coeffs)
        # convert transfer function to state-space representation
        state_space = transfer_func.to_ss()
        # convert continuous state-space to discrete-time with sampling time of 1
        discrete_state_space = state_space.to_discrete(1) # type: ignore

        # the dynamic of the states is x[k+1] = Ax[k] + Bu[k]
        # the output is y[k] = Cx[k]
        self.A = discrete_state_space.A  # state transition matrix
        self.B = discrete_state_space.B  # input matrix
        self.C = discrete_state_space.C  # output matrix

    def init_state_variables(self):
        """
        Initialize the state variables for the system.

        - height_T1: Initial height of Tank 1
        - xprime: Initial state vector
        - flowrate_T1: Initial flowrate of Tank 1
        """
        self.height_T1 = np.asarray([[self.setpoint - 1.]])
        self.xprime = np.asarray([[self.setpoint - 1.]])
        self.flowrate_T1 = (self.C - self.A) / self.B
        self.state_normalizer = self.config.state_normalizer

        self.error_sum = 0
        self.no_of_error = 0
        self.time_step = 0
        self.reward_type = self.config.reward_type

    def init_records(self):
        self.height_T1_record = []
        self.flowrate_T1_record = []
        self.setpoint_T1_record = []
        self.kp_record = []
        self.ti_record = []
        self.reward_record = []

    def inner_step(self, delta_flow_rate: np.ndarray) -> np.ndarray:
        """
        Environment reacts to the inputted action
        """

        # update the flow rate of pump 1 given the change in flow rate
        self.flowrate_T1 += delta_flow_rate[0]
        # bounds the flow rate of pump 1 between 0% and 100%
        self.flowrate_T1 = np.clip(self.flowrate_T1, 0, 100)

        # update the height of tank 1
        xk = self.xprime
        # update the state vector  x[k+1] = Ax[k] + Bu[k]
        self.xprime = xk * self.A + self.flowrate_T1 * self.B
        # update the height of tank 1 y[k] = Cx[k]
        self.height_T1 = xk * self.C
        # clip the height of tank 1 between 0 and 43.1
        self.height_T1 = np.clip(self.height_T1, 0, 43.1)

        # calculates the difference between the current water level and its set
        # point in tanks 1 and 3 store error as old error since it will be
        # updated soon
        self.pid_controller.old_error = self.pid_controller.new_error
        error_T1 = self.setpoint - self.height_T1
        self.no_of_error += 1
        self.error_sum += np.square(error_T1).item()
        self.pid_controller.new_error = error_T1

        self.time_step += 1
        # normalizes the heights and errors and returns them as the
        # environment's state
        return self.get_next_state()

    def get_reward(
        self,
        episode_heights: list[float],
        episode_setpoints: list[float]
    ) -> float:
        base_mse = self.error_sum / (self.no_of_error + 1e-8)
        if self.reward_type == "mse":
            return base_mse / 15

        process = np.array(episode_heights)
        setpoint = np.array(episode_setpoints)
        coeffs = self.config.reward_coeffs

        overshoot = self.percent_overshoot(process, setpoint)
        overshoot_penalty = -coeffs["overshoot_coeff"] * overshoot
        rise_area, settling_area = self.calculate_response_areas(process, setpoint, use_setpoint=True)
        settling_penalty = -coeffs["settling_coeff"] * settling_area
        rise_penalty = -coeffs["rise_coeff"] * rise_area

        mse = -base_mse * coeffs["mse_coeff"]
        return mse + overshoot_penalty + settling_penalty + rise_penalty


    def get_next_state(self) -> np.ndarray:
        return np.asarray([self.setpoint / self.state_normalizer])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.setpoint = self.rng.choice(self.random_sp)
        self.height_T1 = np.array([[self.setpoint - 1.]]) / self.C
        self.xprime = np.array([[self.setpoint - 1.]]) / self.C
        self.flowrate_T1 = (self.C - self.A) / self.B

        return np.array([self.setpoint / self.state_normalizer]), {}

    def percent_overshoot(self, process: np.ndarray, setpoint: np.ndarray) -> float:
        """
        Calculate the percent overshoot of the process.
        Return 0 if no overshoot is found.
        Return the maximum overshoot if multiple overshoots are found.
        """
        overshoot = process - setpoint
        overshoot = overshoot[overshoot > 0]
        if len(overshoot) == 0:
            return 0
        else:
            i = overshoot.argmax()
            sp = setpoint[i] if setpoint.ndim > 0 else setpoint
            return float(overshoot[i] / sp)

    def calculate_response_areas(
        self,
        process: np.ndarray,
        setpoint: Optional[np.ndarray]=None,
        use_setpoint: bool=False,
    ) -> tuple[float, float]:
        """
        Calculate the areas above and below setpoint as percentages of total area.
        Returns (rise_area_percentage, settling_area_percentage)
        """
        assert process.ndim == 1

        if use_setpoint:
            assert setpoint is not None
            target_value = float(setpoint[0] if setpoint.ndim > 0 else setpoint)
        else:
            target_value = process[-1]

        below_setpoint = np.maximum(target_value - process, 0)
        above_setpoint = np.maximum(process - target_value, 0)

        total_deviation = np.sum(below_setpoint) + np.sum(above_setpoint)
        if total_deviation == 0:
            return 0.0, 0.0

        rise_percentage = np.sum(below_setpoint) / total_deviation
        settling_percentage = np.sum(above_setpoint) / total_deviation

        return float(rise_percentage), float(settling_percentage)

    def visualize(self, filename: str = "pvs.gif"):
        if not self.height_T1_record:
            return
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        plt.suptitle('Time Step: 0', y=0.98)

        x = np.arange(self.config.internal_iterations)
        y_max = max(np.max(h) for h in self.height_T1_record) * 1.2
        y_min = 0

        line_height, = ax1.plot([], [], label='Water Level')
        line_setpoint, = ax1.plot([], [], 'r--', label='Setpoint')
        ax1.set_xlim(0, self.config.internal_iterations)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title('Water Level')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Height (cm)')
        ax1.legend()

        scatter = ax2.scatter([], [], c='blue', alpha=0.5)
        ax2.set_xlim(self.config.pid_limits["ti_min"] - 0.1, self.config.pid_limits["ti_max"] + 0.1)
        ax2.set_ylim(self.config.pid_limits["kp_min"] - 0.1, self.config.pid_limits["kp_max"] + 0.1)
        ax2.set_title('PID Parameters')
        ax2.set_xlabel('Ti')
        ax2.set_ylabel('Kp')

        def init():
            line_height.set_data([], [])
            line_setpoint.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
            return line_height, line_setpoint, scatter

        def animate(frame: int):
            actual_frame = frame * 100
            if actual_frame >= len(self.height_T1_record):
                actual_frame = len(self.height_T1_record) - 1

            current_height = self.height_T1_record[actual_frame]
            current_setpoint = self.setpoint_T1_record[actual_frame]

            line_height.set_data(x, current_height)
            line_setpoint.set_data(x, current_setpoint)

            points = np.c_[np.array(np.array(self.ti_record[:actual_frame+1])), self.kp_record[:actual_frame+1]]
            scatter.set_offsets(points)

            plt.suptitle(f'Time Step: {actual_frame}', y=0.98)

            return line_height, line_setpoint, scatter

        num_frames = len(self.height_T1_record) // 100 + 1
        anim = FuncAnimation(
            fig, animate, init_func=init, frames=num_frames,
            interval=1000, blit=False
        )

        writer = PillowWriter(fps=1)
        anim.save(filename, writer=writer)
        plt.close()

    def plot(self, filename: str = "pvs.png"):
        if not self.height_T1_record:
            print("No data to plot")
            return

        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 5, height_ratios=[2, 1])
        ax_water = fig.add_subplot(gs[0, :3])
        ax_reward = fig.add_subplot(gs[0, 3:])
        heights = self.height_T1_record[-1]
        setpoints = self.setpoint_T1_record[-1]
        x = np.arange(len(heights))

        ax_water.plot(x, heights, label='Water Level')
        ax_water.plot(x, setpoints, 'r--', label='Setpoint')
        ax_water.set_title('Final Water Level')
        ax_water.set_xlabel('Time Step')
        ax_water.set_ylabel('Height (cm)')
        ax_water.legend()

        ax_reward.plot(np.arange(len(self.reward_record)), self.reward_record, 'g-', label='Reward')
        ax_reward.set_title('Reward (Final reward: {:.2f})'.format(self.reward_record[-1]))
        ax_reward.set_xlabel('Episodes')
        ax_reward.set_ylabel('Reward')
        ax_reward.legend()

        total_steps = len(self.ti_record)
        steps_per_plot = total_steps // 5

        for i in range(5):
            ax = fig.add_subplot(gs[1, i])
            end_idx = (i + 1) * steps_per_plot

            ax.scatter(
                np.array(self.ti_record[0:end_idx]),
                self.kp_record[0:end_idx],
                c='blue',
                alpha=0.5,
                s=10
            )
            ax.set_xlim(self.config.pid_limits["ti_min"] - 0.1, self.config.pid_limits["ti_max"] + 0.1)
            ax.set_ylim(self.config.pid_limits["kp_min"] - 0.1, self.config.pid_limits["kp_max"] + 0.1)
            ax.set_title(f'Steps {0}-{end_idx}')
            ax.set_xlabel('Ti')
            if i == 0:
                ax.set_ylabel('Kp')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class PIDController:
    """
    PID control

    This class actually only implements the PI controller.
    output = Kp * error + Ki * int error dt

    where:
    - Kp = Proportional gain
    - Ki = Integral gain (Kp/Ti where Ti is integral time)
    """
    def __init__(self, config: PVSConfig):
        self.config = config
        self.initialize_pid_parameters()

    def initialize_pid_parameters(self):
        self.kp = self.config.default_pid["kp"] # proportional gain
        self.ti = self.config.default_pid["ti"] # integral time
        self.kp_max = self.config.pid_limits["kp_max"]
        self.kp_min = self.config.pid_limits["kp_min"]
        self.ti_max = self.config.pid_limits["ti_max"]
        self.ti_min = self.config.pid_limits["ti_min"]

        self.new_error = 0.0
        self.old_error = 0.0


    def update_pid(self, pi_parameters: list[float], KI: bool = False):
        """
        Update the PID parameters

        - pi_parameters: [Kp, Ti]
        """
        self.kp = pi_parameters[0]
        if KI:
            self.ti = 1 / pi_parameters[1]
        else:
            # When KI=False, parameter is interpreted as Ti (integral time)
            self.ti = pi_parameters[1]

        self.kp = np.clip(self.kp, self.kp_min, self.kp_max)
        self.ti = np.clip(self.ti, self.ti_min, self.ti_max)

    def control_output(self) -> np.ndarray:
        # proportional term
        p_term = self.kp * (self.new_error - self.old_error)

        # integral term
        i_term = self.kp * (self.new_error) / (self.ti + 1e-8)

        del_fr_1 = p_term + i_term
        return np.asarray([del_fr_1])

    def reset_pid(self, error_T1: float):
        self.kp = self.config.default_pid["kp"] # proportional gain
        self.ti = self.config.default_pid["ti"] # integral time
        self.new_error = error_T1


class PVSChangeAction(BasePVSEnv):
    """RL environment implementation"""
    def __init__(
        self,
        cfg: PVSConfig,
    ):
        BasePVSEnv.__init__(self, cfg)

        self.prev_pid_params = np.array([
            self.config.default_pid["kp"],
            self.config.default_pid["ti"]
        ])

        self.curr_step = 0
        self.reset_temperature = self.config.reset_temperature
        self.last_reset_time = 0
        self.reset_buffer_pointer = 0
        self.no_reset = self.config.no_reset

        self.init_spaces()
        if self.config.use_reset_buffer:
            self.init_reset_buffer()

    def init_spaces(self):
        limits = self.config.pid_limits
        self.observation_space = spaces.Box(
            low=np.array([limits["kp_min"], limits["ti_min"]]),
            high=np.array([limits["kp_max"], limits["ti_max"]]),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1]),
            high=np.array([0.1, 0.1]),
            shape=(2,),
            dtype=np.float32,
        )

    def init_reset_buffer(self):
        size = (self.config.reset_buffer_size, 2)
        self._reset_buffer = np.empty(size)
        self._reset_buffer_priorities = np.empty(self.config.reset_buffer_size)
        self._reset_buffer_priorities[:] = -np.inf

        self.tracking_threshold = -np.inf
        self.start_state = np.array([
            self.config.default_pid["kp"],
            self.config.default_pid["ti"]
        ])
        self.start_state_r = self._step(self.start_state, False)[1]
        self.tracking_threshold = int(self.start_state_r) - 1

        self._reset_buffer[0, :] = self.start_state
        self._reset_buffer_priorities[0] = self.start_state_r
        self.reset_buffer_pointer = 0
        self.kp_record = []
        self.ti_record = []
        self.height_T1_record = []
        self.flowrate_T1_record = []
        self.setpoint_T1_record = []

    def get_next_pid_params(self, a: np.ndarray) -> np.ndarray:
        pid_params = np.array([
            self.prev_pid_params[0] + a[0],
            self.prev_pid_params[1] + a[1]
        ])
        return self.pid_param_clip(pid_params)

    def pid_param_clip(self, pid_params: np.ndarray) -> np.ndarray:
        limits = self.config.pid_limits

        pid_params[0] = np.clip(pid_params[0], limits["kp_min"], limits["kp_max"])
        pid_params[1] = np.clip(
            pid_params[1], limits["ti_min"], limits["ti_max"]
        )
        return pid_params

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        self.curr_step += 1

        pid_params = self.get_next_pid_params(action)
        return self._step(pid_params)

    def add_to_buffer(self, r: float, pid_params: np.ndarray):
        if self.reset_buffer_pointer >= self.config.reset_buffer_size:
            self.reset_buffer_pointer = 0  # wrap around to start
        self._reset_buffer[self.reset_buffer_pointer, :] = pid_params
        self._reset_buffer_priorities[self.reset_buffer_pointer] = r
        self.reset_buffer_pointer += 1

    @property
    def _reset_prob(self) -> np.ndarray:
        # get priorities/rewards for states in the buffer up to current pointer
        p = self._reset_buffer_priorities[:self.reset_buffer_pointer]
        if len(p) == 0:
            return np.array([1.0])

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
    def _reset_ind(self) -> int:
        p = self._reset_prob
        return np.random.choice(p.shape[0], p=p)

    @property
    def _reset_state(self) -> np.ndarray:
        ind = self._reset_ind
        return self._reset_buffer[ind, :]


    def _step(
        self, pid_params: np.ndarray, add: bool=True
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # interact with the environment for a number of steps
        self.pid_controller.update_pid(pid_params.tolist(), KI=False)
        # Record the actual PID parameters used, not the raw actions
        self.kp_record.append(self.pid_controller.kp)
        self.ti_record.append(self.pid_controller.ti)
        episode_heights = []
        episode_flows = []
        episode_setpoints = []
        self.error_sum = 0
        self.no_of_error = 0

        for _ in range(self.config.internal_iterations):
            self.inner_step(self.pid_controller.control_output())
            episode_heights.append(self.height_T1.item())
            episode_flows.append(self.flowrate_T1.item())
            episode_setpoints.append(self.setpoint)

        reward = self.get_reward(episode_heights, episode_setpoints)

        # check if the process is unstable
        overfill = np.any(np.array(episode_heights) > np.array(episode_setpoints))
        severe_overfill = np.any(np.array(episode_heights) > 1.075 * np.array(episode_setpoints))
        unstable_params = bool(overfill or severe_overfill or reward < self.tracking_threshold)
        # reset if unstable or if the PID parameters are negative
        if (unstable_params or np.any(pid_params < 0)) and not self.no_reset:
            if self.config.use_reset_buffer:
                pid_params = self._reset_state
            else:
                pid_params = np.array([
                    self.config.default_pid["kp"],
                    self.config.default_pid["ti"]
                ])
            print("Resetting PID parameters to default", pid_params)
            reward -= 10
        elif add and self.config.use_reset_buffer:
            self.add_to_buffer(reward, self.prev_pid_params)

        s_next = pid_params
        self.prev_pid_params = pid_params

        setpoint = np.array(episode_setpoints)
        process = np.array(episode_heights)
        self.height_T1_record.append(np.array(episode_heights))
        self.flowrate_T1_record.append(np.array(episode_flows))
        self.setpoint_T1_record.append(np.array(episode_setpoints))

        rise_area, settling_area = self.calculate_response_areas(process, setpoint, use_setpoint=True)
        info = {
            "unstable": unstable_params,
            "percent_overshoot": self.percent_overshoot(process, setpoint),
            "rise_time": rise_area,
            "settling_time": settling_area,
            "environment_pid": pid_params,
        }

        self.setpoint = self.rng.choice(self.random_sp)
        self.height_T1 = np.array([[self.setpoint - 1.]]) / self.C
        self.xprime = np.array([[self.setpoint - 1.]]) / self.C
        self.flowrate_T1 = (self.C - self.A) / self.B

        error_T1 = self.setpoint - self.height_T1
        self.error_sum += np.square(error_T1).item()
        self.no_of_error += 1
        self.pid_controller.reset_pid(error_T1)
        self.reward_record.append(reward)
        return s_next, reward, False, False, info


    def reset(
        self,
        *,
        seed: Optional[int]=None,
        options: Optional[dict]=None,
    ) -> tuple[np.ndarray, dict]:
        super(BasePVSEnv, self).reset(seed=seed, options=options)
        return self.prev_pid_params, {}


gym.register(
    id='PVS-v0',
    entry_point='corerl.environment.pvs:PVSChangeAction',
    kwargs={'cfg': PVSConfig()}
)
