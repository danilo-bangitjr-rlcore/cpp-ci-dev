from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from .flows import FlowModel as FM
from .flows import SensorModel


class TankModel:
    # Model of the biochemical processes in an activated sludge tank system
    # using the ASM1 model. All parameters are taken from the BSM1 paper
    # http://iwa-mia.org/wp-content/uploads/2018/01/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf

    def __init__(self, sensor_list: tuple["SensorModel", ...] = (), param_override: dict | None = None):
        # Initialize the tank model with a list of sensors
        #   sensor_list: Tuple of SensorModel objects for the tank sensors
        #   param_override: Optional dictionary to override default tank parameters
        if param_override is None:
            param_override = {}
        self.tank_params = ASM1Params(**param_override)
        self.tank_flow = FM()
        self.sensors = sensor_list

    def step(self, influent: FM, aeration: float = 0., dt: float = 1. / 60 / 24) -> list[float]:
        # Step the tank model for a given influent flow, aeration rate, and time step
        #   influent: FlowModel containing the influent flow and concentrations
        #   aeration: Aeration rate for the tank (units d^-1)
        #   dt: Time step in days (default 15 minutes)

        # Mass balance requires tank flow = influent flow
        self.tank_flow.Q = influent.Q

        if self.tank_params.integrator == 'Euler':
        # Euler step the concentrations
            dX = self._calc_derivatives(0.0, self.tank_flow.X, self.tank_flow.Q, influent.X, aeration)
            self.tank_flow.X += dX * dt
        elif self.tank_params.integrator == 'RK45':
            # RK45 step for more accurate integration
            sol = solve_ivp(self._calc_derivatives, (0, dt), self.tank_flow.X, t_eval=[dt], \
                            method='RK45', rtol=1e-3, atol=1e-3, args = (self.tank_flow.Q, influent.X, aeration))

            self.tank_flow.X = sol.y[:,-1]

        # Ensure the concentrations are non-negative
        self.tank_flow.X = np.clip(self.tank_flow.X, 0, None)

        return [s.step(self.tank_flow) for s in self.sensors]

    def _calc_derivatives(self, _: float, tank_flow_X: np.ndarray, tank_flow_Q: float,
                          influent_X: np.ndarray, aeration: float) -> np.ndarray:
        # print(f'Tank Flow: {tank_flow_X}')
        # Calculate the concentration delta based on influent, internal reactions, and effluent
        #   Reactions + Influent - Effluent
        dX = self._calculate_reaction_rates(tank_flow_X) + \
             tank_flow_Q * (influent_X - tank_flow_X) / self.tank_params.volume

        #   Aeration (affects the oxygen concentration)
        dX[FM.state_idx['SO']] += aeration * (8 - tank_flow_X[FM.state_idx['SO']])
        # print(f'dx {dX}')
        return dX

    def reset(self) -> list[float]:
        # Reset the tank model to its initial state
        self.tank_flow.reset()
        return [s.reset(self.tank_flow) for s in self.sensors]

    def _calculate_processes(self, flow_X: np.ndarray) -> np.ndarray:
        # Section 2.2.2 of the BSM1 paper
        rho = np.zeros(8)

        # j = 1: Aerobic growth of heterotrophs
        rho[0] = self.tank_params.mu_H * flow_X[FM.state_idx['SS']]  / \
                    (self.tank_params.K_S + flow_X[FM.state_idx['SS']]) * \
                    flow_X[FM.state_idx['SO']] / \
                    (self.tank_params.K_O_H + flow_X[FM.state_idx['SO']]) * \
                    flow_X[FM.state_idx['XBH']]

        # j = 2: Anoxic growth of heterotrophs
        rho[1] = self.tank_params.mu_H * flow_X[FM.state_idx['SS']] / \
                 (self.tank_params.K_S + flow_X[FM.state_idx['SS']]) * \
                 self.tank_params.K_O_H / \
                 (self.tank_params.K_O_H + flow_X[FM.state_idx['SO']]) * \
                 flow_X[FM.state_idx['SNO']] / \
                 (self.tank_params.K_NO + flow_X[FM.state_idx['SNO']]) * \
                 self.tank_params.eta_g * flow_X[FM.state_idx['XBH']]

        # j = 3: Aerobic growth of autotrophs
        rho[2] = self.tank_params.mu_A * flow_X[FM.state_idx['SNH']] / \
                 (self.tank_params.K_NH + flow_X[FM.state_idx['SNH']]) * \
                 flow_X[FM.state_idx['SO']] / \
                 (self.tank_params.K_O_A + flow_X[FM.state_idx['SO']]) * \
                 flow_X[FM.state_idx['XBA']]

        # j = 4: Decay of heterotrophs
        rho[3] = self.tank_params.b_H * flow_X[FM.state_idx['XBH']]

        # j = 5: Decay of autotrophs
        rho[4] = self.tank_params.b_A * flow_X[FM.state_idx['XBA']]

        # j = 6: Ammonification of soluble organic nitrogen
        rho[5] = self.tank_params.k_a * flow_X[FM.state_idx['SND']] * \
                 flow_X[FM.state_idx['XBH']]

        # j = 7: Hydrolysis of entrapped organics
        XS_over_XBH = flow_X[FM.state_idx['XS']] / flow_X[FM.state_idx['XBH']]
        rho[6] = self.tank_params.k_h * XS_over_XBH / \
                 (self.tank_params.K_X + XS_over_XBH) * \
                 (flow_X[FM.state_idx['SO']] / \
                  (self.tank_params.K_O_H + flow_X[FM.state_idx['SO']]) + \
                  self.tank_params.eta_h * self.tank_params.K_O_H / \
                  (self.tank_params.K_O_H + flow_X[FM.state_idx['SO']]) * \
                  flow_X[FM.state_idx['SNO']] / \
                  (self.tank_params.K_NO + flow_X[FM.state_idx['SNO']])) * \
                 flow_X[FM.state_idx['XBH']]

        # j = 8: Hydrolisis of entrapped organic nitrogen
        rho[7] = rho[6] * flow_X[FM.state_idx['XND']] / flow_X[FM.state_idx['XS']]

        return rho

    def _calculate_reaction_rates(self, flow_X: np.ndarray) -> np.ndarray:
        # Section 2.2.3 of the BSM1 paper
        rho = self._calculate_processes(flow_X)

        # Calculate reaction rates for each of the biochemical components
        rates = np.zeros(FM.nX)

        # k = 1: Inert organic matter => 0
        rates[FM.state_idx['SI']] = 0

        # k = 2: Readily biodegradable substrate
        rates[FM.state_idx['SS']] = -1 / self.tank_params.Y_H * ( rho[0] + rho[1] ) + rho[6]

        # k = 3: Particulate inert organic matter
        rates[FM.state_idx['XI']] = 0

        # k = 4: Slowly biodegradable substrate
        rates[FM.state_idx['XS']] = (1 - self.tank_params.f_P) * (rho[3] + rho[4]) - rho[6]

        # k = 5: Active heterotrophic biomass XB,H
        rates[FM.state_idx['XBH']] = rho[0] + rho[1] - rho[3]

        # k = 6: Active autotrophic biomass XB,A
        rates[FM.state_idx['XBA']] = rho[2] - rho[4]

        # k = 7: Particulate products arising from biomass decay XP
        rates[FM.state_idx['XP']] = self.tank_params.f_P * (rho[3] + rho[4])

        # k = 8: Oxygen SO
        rates[FM.state_idx['SO']] = -(1 - self.tank_params.Y_H) / self.tank_params.Y_H * rho[0] - \
                                    (4.57 - self.tank_params.Y_A) / self.tank_params.Y_A * rho[2]

        # k = 9: Nitrate and nitrite nitrogen SNO
        rates[FM.state_idx['SNO']] = -(1 - self.tank_params.Y_H) / (2.86 * self.tank_params.Y_H) * rho[1] + \
                                     1 / self.tank_params.Y_A * rho[2]

        # k = 10: NH 4+ + NH 3 nitrogen SNH
        rates[FM.state_idx['SNH']] = -self.tank_params.i_XB * (rho[0] + rho[1]) - \
                                     (self.tank_params.i_XB + 1 / self.tank_params.Y_A) * rho[2] + rho[5]

        # k = 11: Soluble biodegradable organic nitrogen SND
        rates[FM.state_idx['SND']] = -rho[5] + rho[7]

        # k = 12: Particulate biodegradable organic nitrogen XND
        rates[FM.state_idx['XND']] = (self.tank_params.i_XB - self.tank_params.f_P*self.tank_params.i_XP) * \
                                     (rho[3] + rho[4]) - rho[7]

        # k = 13: Alkalinity
        rates[FM.state_idx['SALK']] = -self.tank_params.i_XB / 14 * rho[0] + \
                                      ((1 - self.tank_params.Y_H) / (14 * 2.86 * self.tank_params.Y_H) - \
                                       self.tank_params.i_XB / 14) * rho[1] - \
                                      (self.tank_params.i_XB / 14 + 1 / 7 / self.tank_params.Y_A) * rho[2] + \
                                      1 / 14 * rho[5]

        return rates

@dataclass
class ASM1Params:
    volume: float = 1000.0

    # Stoichiometric parameters
    Y_A = 0.24 # g cell COD formed.(g N oxidized)^-1
    Y_H = 0.67 # g cell COD formed.(g COD oxidized)^-1
    f_P = 0.08 # dimensionless
    i_XB = 0.08 # g N.(g COD)^-1 in biomass
    i_XP = 0.06 # g N.(g COD)^-1 in particulate product

    # Kinetic parameters
    mu_H = 4.0 # d^-1
    K_S = 10.0 # g COD.m^-3
    K_O_H = 0.2 # g (-COD).m^-3
    K_NO = 0.5 #  g NO3-N.m^-3
    b_H = 0.3 # d^-1
    eta_g = 0.8 # dimensionless
    eta_h = 0.8 # dimensionless
    k_h = 3.0 # g slowly biodegradable COD.(g cell COD.d)^-1
    K_X = 0.1 # g slowly biodegradable COD.(g cell COD)^-1
    mu_A = 0.5 # d^-1
    K_NH = 1.0 # g NH3-N.m^-3
    b_A = 0.05 # d^-1
    K_O_A = 0.4 # g (-COD).m^-3
    k_a = 0.05 # m^3 (g COD.d)^-1

    # Integrator
    integrator: Literal['Euler', 'RK45'] = 'Euler'
