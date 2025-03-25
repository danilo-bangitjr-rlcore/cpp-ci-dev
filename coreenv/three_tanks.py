from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from corerl.configs.config import config
from corerl.configs.loader import config_from_dict


@config()
class ThreeTankConfig:
    steps_between_target_updates: int | None = 30  # Number of steps between target updates
                                                    #   None: no updates
                                                    #   <=1: updates every step
                                                    #   2: updates every other step
    target_filter_alpha: float = 0.9 # Filter the target values to ensure smooth updates
                                     #    0: No filtering
                                     #    1: Target won't change

@dataclass
class ThreeTankConstants:
    # These constants come from the paper linked below
    g: float = 983.991  # gravitational constant [cm/s^2]
    A_T: float = 180.54 # tank x-sectional area [cm^2]
    A_V: float = 0.385   # valve x-sectional area [cm^2]
    H_Max: float = 43.1 # Max water height of the tanks [cm]
                        # Note: this was not in the paper, we chose a reasonable value
    H_V12: float = 30.6 # Height of valve 1 and 2 from the tank bottom [cm]
    H_V34: float = 15.3 # Height of valve 3 and 4 from the tank bottom [cm]
    C_V12: float = 0.26 # Drain coefficient of valve 1 and 2
    C_V3: float = 0.29  # Drain coefficient of valve 3
    C_V4: float = 0.33  # Drain coefficient of valve 4
    C_V5: float = 0.78  # Drain coefficient of valve 5
    C_V7: float = 0.69  # Drain coefficient of valve 7
    C_V9: float = 0.82  # Drain coefficient of valve 9
    K_1: float = 1.47   # Pump 1 coefficient
    K_2: float = 1.435  # Pump 2 coefficient
    dt: float = 5       # Time between updates [s]

class ThreeTankEnv(gym.Env):
    """
    A Three Tank Environment with PID Controller.
    Taken from: https://www.sciencedirect.com/science/article/pii/S0959152421000950#sec4
    The flow from pumps 1 and 2 is controlled in order to track target heights in
    tanks 1 and 3 respectively. The 3 tanks are connected with level-dependent connections
    and are each connected to a central drain reservoir

         ┌───────┐
         │Pump 1 │
         └───┬───┘
             │
   ┌─┐  ┌────┴────┐
   │R├──┤ Tank 1  │
   │e│ ┌┴────┬──┬─┘
   │s│ │     │  │
   │e│ └┬────┴──┴─┐
   │r├──┤ Tank 2  │
   │v│ ┌┴────┬──┬─┘
   │o│ │     │  │
   │i│ └┬────┴──┴─┐
   │r├──┤ Tank 3  │
   └─┘  └────┬────┘
             │
         ┌───┴───┐
         │Pump 2 │
         └───────┘

    States: [H_1, H_2, H_3, H1_SP, H3_SP] (Tank heights and setpoint heights)
    Actions: [p_1, p_2] (Pump flowrates)
    """
    def __init__(self, cfg: ThreeTankConfig | None = None):
        if  cfg is None:
            cfg = ThreeTankConfig()

        super().__init__()
        self.cfg = cfg
        self.constants = ThreeTankConstants()

        self.H = np.zeros(3) # Tank heights for tanks 1,2, and 3
        self.H_t = np.zeros(2) # Time-varying targets for tank 1 and 3 heights
        self.H_t_unfilt = np.zeros(2)
        self.target_counter = self.cfg.steps_between_target_updates

        self.action_space = gym.spaces.Box(np.zeros(2), np.ones(2) * 100, dtype=np.float64)
        self.observation_space = gym.spaces.Box(np.zeros(5), np.ones(5) * self.constants.H_Max, dtype=np.float64)

        self.history = {
            'H1': [], 'H3': [],
            'T1': [], 'T3': [],
            'P1': [], 'P2': [],
            'time': []
        }
        self.current_step = 0

    def step(self, action: np.ndarray):
        # Calculate the next step
        self.current_step += 1
        self.history['H1'].append(self.H[0])
        self.history['H3'].append(self.H[2])
        self.history['T1'].append(self.H_t[0])
        self.history['T3'].append(self.H_t[1])
        self.history['P1'].append(action[0])
        self.history['P2'].append(action[1])
        self.history['time'].append(self.current_step)

        self.update_tank_heights(action)
        reward = self.calc_reward()

        self.update_target_heights()

        return np.hstack([self.H,self.H_t]), reward, False, False, {'history': self.history}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self.H[:] = self.np_random.random(3) * self.constants.H_Max
        self.H_t[:] = self.np_random.random(2) * self.constants.H_Max
        self.H_t_unfilt[:] = self.H_t
        self.target_counter = self.cfg.steps_between_target_updates

        self.history = {
            'H1': [], 'H3': [],
            'T1': [], 'T3': [],
            'P1': [], 'P2': [],
            'time': []
        }
        self.current_step = 0

        return np.hstack([self.H,self.H_t]), {}

    def update_tank_heights(self, action: np.ndarray):
        # These equations come from section 4 of the above paper
        c = self.constants
        H_1, H_2, H_3 = self.H
        P_1, P_2 = action
        def delta_valve(H_a: float, H_b: float, H_v: float):
            # Difference in tank height above a given valve height
            return max( H_a - H_v, 0) - max( H_b - H_v, 0)
        def Q_valve(C: float, dh: float):
            # Valve flow given a difference in height across the valve
            return C * c.A_V * np.sign(dh) * np.sqrt(2 * c.g * np.abs(dh))

        # Flow through the pumps
        Q_P1 = c.K_1 * P_1
        Q_P2 = c.K_2 * P_2
        # Flow through each of the valves (see Fig. 4 in the paper)
        Q_V1 = Q_valve(c.C_V12, delta_valve(H_2, H_1, c.H_V12))
        Q_V2 = Q_valve(c.C_V12, delta_valve(H_2, H_3, c.H_V12))
        Q_V3 = Q_valve(c.C_V3,  delta_valve(H_2, H_1, c.H_V34))
        Q_V4 = Q_valve(c.C_V4,  delta_valve(H_2, H_3, c.H_V34))
        Q_V5 = -c.C_V5 * c.A_V * np.sqrt(2 * c.g * H_1)
        Q_V7 = -c.C_V7 * c.A_V * np.sqrt(2 * c.g * H_2)
        Q_V9 = -c.C_V9 * c.A_V * np.sqrt(2 * c.g * H_3)
        # Cumulative flow into each tank
        Q_T1 =  Q_V1 + Q_V3 + Q_V5 + Q_P1
        Q_T2 = -Q_V1 - Q_V2 - Q_V3 - Q_V4 + Q_V7
        Q_T3 =  Q_V2 + Q_V4 + Q_V9 + Q_P2

        # Update each tank height based on cumulative flow
        self.H[0] += c.dt * Q_T1 / c.A_T
        self.H[1] += c.dt * Q_T2 / c.A_T
        self.H[2] += c.dt * Q_T3 / c.A_T

        # Clip the heights based on the max height of the tanks
        # Note: This is not in the paper, but we assume the presence
        # of a large overflow drain at the top of the tanks
        self.H[:] = np.clip(self.H, np.zeros(3), np.ones(3) * c.H_Max)

    def update_target_heights(self):
        if self.target_counter is not None:
            # Update the setpoints at a fixed interval
            self.target_counter = self.target_counter - 1

            if self.target_counter <= 0:
                self.H_t_unfilt = self.np_random.random(2) * self.constants.H_Max
                self.target_counter = self.cfg.steps_between_target_updates

        a = self.cfg.target_filter_alpha
        self.H_t = a * self.H_t + (1 - a) * self.H_t_unfilt

    def calc_reward(self):
        error = -((self.H[0] - self.H_t[0])**2 + (self.H[2] - self.H_t[1])**2)
        return error/(2*self.constants.H_Max**2)

    def plot(self, filename: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(self.history['time'], self.history['H1'], 'b-', label='Tank 1 Height')
        ax1.plot(self.history['time'], self.history['H3'], 'g-', label='Tank 3 Height')
        ax1.plot(self.history['time'], self.history['T1'], 'b--', label='Tank 1 Target')
        ax1.plot(self.history['time'], self.history['T3'], 'g--', label='Tank 3 Target')
        ax1.set_ylabel('Height (cm)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history['time'], self.history['P1'], 'r-', label='Pump 1')
        ax2.plot(self.history['time'], self.history['P2'], 'm-', label='Pump 2')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Pump Flow Rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(filename)


gym.register(
    id='ThreeTank-v1',
    entry_point='corerl.environment.three_tanks:ThreeTankEnv'
)
