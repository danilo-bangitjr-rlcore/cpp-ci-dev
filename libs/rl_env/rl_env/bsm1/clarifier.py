# Clarifier model for the BSM1 environment. Model parameters
# are taken from the BSM1 paper
# http://iwa-mia.org/wp-content/uploads/2018/01/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from .flows import FlowModel, SbhSensorModel, SensorModel


class ClarifierModel:
    def __init__(self, sensor_list: tuple["SensorModel", ...] = (),
                       sludge_bed_height_sensor: SbhSensorModel | None = None):
        self.params = ClarifierParams()

        self.effluent_flow = FlowModel()
        self.under_flow = FlowModel()
        self.sensors = sensor_list
        self.sludge_bed_height_sensor = sludge_bed_height_sensor

        X_f_components = ['XS', 'XP', 'XI', 'XBH', 'XBA']
        self.X_f_indices = [FlowModel.state_idx[c] for c in X_f_components]
        solids_components = [*X_f_components, 'XND', 'TSS']
        self.solids_indices = [FlowModel.state_idx[c] for c in solids_components]
        soluble_components = ['SI', 'SS', 'SO', 'SNO', 'SNH', 'SND', 'SALK']
        self.soluble_indices = [FlowModel.state_idx[c] for c in soluble_components]
        self.n_Z = len(soluble_components)

        # Initialize the internal variables: Sludge Concentration and Soluble Concentrations
        initial_flow = FlowModel()
        self.X_SC = np.ones(self.params.n_layers) * 0.75 * np.sum(initial_flow.X[self.X_f_indices])
        self.Z_SC = np.tile(initial_flow.X[self.soluble_indices],[self.params.n_layers,1])

    def step(self, influent: FlowModel, underflow_Q: float, dt: float = 1. / 60 / 24):
        effluent_Q = influent.Q - underflow_Q
        v_up = effluent_Q/self.params.area
        v_dn = underflow_Q/self.params.area
        X_f = 0.75 * np.sum(influent.X[self.X_f_indices])
        Q_f_Z_f = influent.Q * influent.X[self.soluble_indices]

        # Integrate internal state
        if self.params.integrator == 'Euler':
            dX = self.sludge_settling_model(self.X_SC, influent.Q, X_f, v_dn, v_up)
            dZ = self.soluble_model(self.Z_SC, Q_f_Z_f, v_dn, v_up)
            self.X_SC += dX * dt
            self.Z_SC += dZ * dt

        elif self.params.integrator == 'RK45':
            sol = solve_ivp(self._calc_derivatives, (0, dt), np.hstack([self.X_SC,self.Z_SC.flatten()]), t_eval=[dt], \
                            method='RK45', rtol=1e-3, atol=1e-3, args=(influent.Q, X_f, v_dn, v_up, Q_f_Z_f))

            # Update the sludge concentration in each layer
            self.X_SC = sol.y[:len(self.X_SC), -1]

            # Update the soluble concentrations in each layer
            self.Z_SC = np.reshape(sol.y[len(self.X_SC):, -1], self.Z_SC.shape)

        self.X_SC = np.clip(self.X_SC, 0, None)
        self.Z_SC = np.clip(self.Z_SC, 0, None)

        # Update effluent and underflow flows
        self.effluent_flow.X[self.solids_indices] = self.X_SC[-1] / X_f * influent.X[self.solids_indices]
        self.under_flow.X[self.solids_indices] = self.X_SC[0] / X_f * influent.X[self.solids_indices]

        self.effluent_flow.X[FlowModel.state_idx['TSS']] = 0.75*np.sum(self.effluent_flow.X[self.X_f_indices])
        self.under_flow.X[FlowModel.state_idx['TSS']] = 0.75*np.sum(self.under_flow.X[self.X_f_indices])

        self.effluent_flow.X[self.soluble_indices] = self.Z_SC[-1]
        self.under_flow.X[self.soluble_indices] = self.Z_SC[0]

        self.effluent_flow.Q = effluent_Q
        self.under_flow.Q = underflow_Q

        if self.sludge_bed_height_sensor is None:
            sbh_val = []
        else:
            sbh_val = [self.sludge_bed_height_sensor.step(self.X_SC, self.params.layer_height)]

        return [s.step(self.effluent_flow) for s in self.sensors] + sbh_val

    def _calc_derivatives(self, _: float, y: np.ndarray, influent_Q: float, X_f: float,
                          v_dn: float, v_up: float, Q_f_Z_f: np.ndarray):
        X_SC = y[:len(self.X_SC)]
        Z_SC = np.reshape(y[len(self.X_SC):],self.Z_SC.shape)

        dX = self.sludge_settling_model(X_SC, influent_Q, X_f, v_dn, v_up)
        dZ = self.soluble_model(Z_SC, Q_f_Z_f, v_dn, v_up)
        return np.hstack([dX,dZ.flatten()])

    def reset(self):
        self.effluent_flow.reset()
        self.under_flow.reset()
        initial_flow = FlowModel()
        self.X_SC = np.ones(self.params.n_layers) * 0.75 * np.sum(initial_flow.X[self.X_f_indices])
        self.Z_SC = np.tile(initial_flow.X[self.soluble_indices],[self.params.n_layers,1])

        if self.sludge_bed_height_sensor is None:
            sbh_val = []
        else:
            sbh_val = [self.sludge_bed_height_sensor.reset(self.X_SC, self.params.layer_height)]

        return [s.reset(self.effluent_flow) for s in self.sensors] + sbh_val

    def sludge_settling_model(self, X_SC: np.ndarray, Q_f: float, X_f: float, v_dn: float, v_up: float) -> np.ndarray:
        # Sludge settling model based on the BSM1 paper
        # Q_f: Flow rate of the influent
        # X_f: Concentration of the influent
        # v_dn: Downward velocity in the clarifier
        # v_up: Upward velocity in the clarifier

        # Calculate solid flux due to gravity JS
        X_min = self.params.nonsettleable_fraction * X_f
        v_s = np.zeros(self.params.n_layers)
        J_sc = np.zeros(self.params.n_layers)
        for i in range(self.params.n_layers):
            v_s[i] = np.clip(self.params.vesilind_settling_velocity * \
                             (np.exp(-self.params.hindered_zone_settling * (X_SC[i] - X_min)) - \
                              np.exp(-self.params.flocculant_zone_settling * (X_SC[i] - X_min))), \
                             0, self.params.max_settling_velocity)
            if i >= self.params.inflow_layer:
                if X_SC[i-1] > self.params.threshold_concentration:
                    J_sc[i] = min(v_s[i]*X_SC[i], v_s[i-1]*X_SC[i-1])
                else:
                    J_sc[i] = v_s[i] * X_SC[i]
        J_s = v_s * X_SC

        dX = np.zeros(self.params.n_layers)
        dX[0] = v_dn * (X_SC[1] - X_SC[0]) + min(J_s[0], J_s[1])
        for i in range(1, self.params.n_layers-1):
            if i < self.params.inflow_layer:
                dX[i] = v_dn * (X_SC[i+1] - X_SC[i]) + min(J_s[i], J_s[i+1]) - min(J_s[i-1], J_s[i])
            elif i == self.params.inflow_layer:
                dX[i] = Q_f * X_f / self.params.area + J_sc[i+1] - (v_up + v_dn) * X_SC[i] - min(J_s[i-1], J_s[i])
            else:
                dX[i] = v_up * (X_SC[i-1] - X_SC[i]) + J_sc[i+1] - J_sc[i]
        dX[-1] = v_up * (X_SC[-2] - X_SC[-1]) - J_sc[-1]
        dX /= self.params.layer_height

        return dX

    def soluble_model(self, Z_SC: np.ndarray, Q_f_Z_f: np.ndarray, v_dn: float, v_up: float) -> np.ndarray:
        # Soluble model based on the BSM1 paper
        # Q_f_Z_f: Flow rate of the influent multiplied by the soluble concentrations
        # v_dn: Downward velocity in the clarifier
        # v_up: Upward velocity in the clarifier

        dZ = np.zeros([self.params.n_layers, self.n_Z])
        for i in range(self.params.n_layers):
            if i < self.params.inflow_layer:
                dZ[i,:] = v_dn * (Z_SC[i+1,:] - Z_SC[i,:])
            elif i == self.params.inflow_layer:
                dZ[i,:] = Q_f_Z_f/self.params.area - (v_dn + v_up) * Z_SC[i,:]
            else:
                dZ[i,:] = v_up * (Z_SC[i-1,:] - Z_SC[i,:])

        dZ /= self.params.layer_height

        return dZ

@dataclass
class ClarifierParams:
    # Clarifier parameters
    area: float = 1500.0
    layer_height: float = 0.4
    n_layers: int = 10
    inflow_layer: int = 5 # Layer 6 with 1 indexing

    # Settling parameters
    max_settling_velocity: float = 250.0
    vesilind_settling_velocity: float = 474.0
    hindered_zone_settling: float = 0.000576
    flocculant_zone_settling: float = 0.00286
    nonsettleable_fraction: float = 0.00228
    threshold_concentration: float = 3000.0

    # Integrator
    integrator: Literal['Euler', 'RK45'] = 'Euler'
