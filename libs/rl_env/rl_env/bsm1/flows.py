# Flow, influent and sensor models for the BSM1 environment
import csv
from importlib import resources
from typing import Literal

import numpy as np

from rl_env import bsm1


class FlowModel:
    state_names = ['SI',   'SS',  'XI',  'XS',  'XBH', 'XBA',  'XP',
                   'SO',   'SNO', 'SNH', 'SND', 'XND', 'SALK', 'TSS',
                   'TEMP']
    state_idx = {n: i for i, n in enumerate(state_names)}
    nX = len(state_names)
    # Use the average influent values from the paper as the default initial state
    _default_state = np.array([30, 69.5, 51.2, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95, 10.59, 7, 211.2675, 15])

    def __init__(self, X: np.ndarray | None = None, Q: float = 0):
        if X is None:
            X = self._default_state.copy()
        assert len(X) == self.nX, 'Incorrect state array size'

        self.X = X # Flow state: concentrations (x13), TSS, temperature
        self.Q = Q # Flow rate

    def __add__(self, other:"FlowModel")->"FlowModel":
        combined_Q = self.Q + other.Q
        combined_X = (self.X*self.Q + other.X*other.Q)/combined_Q
        return FlowModel(combined_X, combined_Q)

    def reset(self) -> None:
        self.X[:] = self._default_state
        self.Q = 0

    def get(self, name: 'str')->float:
        if name == 'Q':
            return self.Q
        return self.X[self.state_idx[name]]

    def copy(self)->"FlowModel":
        return FlowModel(self.X.copy(),self.Q)

    def copy_from(self, other:"FlowModel"):
        self.X[:] = other.X
        self.Q = other.Q

class InfluentModel:
    def __init__(self, sensor_list: tuple["SensorModel", ...] = ()):
        # Take the steady state values from the paper
        self.influent_flow = FlowModel()
        self.sensors = sensor_list

    def step(self, dt: float) -> list[float]:
        # Step the sensors to get the current values
        return [s.step(self.influent_flow) for s in self.sensors]

    def reset(self) -> list[float]:
        return [s.reset(self.influent_flow) for s in self.sensors]

class SteadyStateInfluent(InfluentModel):
    def __init__(self, sensor_list: tuple["SensorModel", ...] = ()):
        super().__init__(sensor_list)
        # Set the steady state values from the paper
        self.steady_state_inf = np.array([30.0, 69.5, 51.2, 202.32, 28.17, 0.0, 0.0, 0.0,
                                        0.0, 31.56, 6.95, 10.59, 7.0, 211.2675, 15.0], dtype=float)
        self.influent_flow.X = self.steady_state_inf  # Steady state flow
        self.influent_flow.Q = 18446

    def reset(self) -> list[float]:
        self.influent_flow.X = self.steady_state_inf  # Steady state flow
        self.influent_flow.Q = 18446
        return [s.reset(self.influent_flow) for s in self.sensors]

class TimeSeriesInfluent(InfluentModel):
    def __init__(self, file_name: str, sensor_list: tuple["SensorModel", ...] = ()):
        super().__init__(sensor_list)

        with resources.open_text(bsm1, "influent_data/" + file_name) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            data = np.array(list(reader))

        self.series_t = data[:,0]
        x_inds = [*list(range(1, 15)), 16]
        self.series_X = data[:,x_inds]
        self.series_Q = data[:,15]
        self.t = 0
        t_diff = np.median(np.diff(self.series_t))
        self.t_max = np.max(self.series_t) + t_diff/2

    def step(self, dt: float) -> list[float]:
        t_ind = np.argmin(np.abs(self.series_t - self.t))
        self.influent_flow.X = self.series_X[t_ind,:]
        self.influent_flow.Q = self.series_Q[t_ind]

        self.t += dt
        if self.t > self.t_max:
            self.t -= 1.0

        return super().step(dt)

    def reset(self) -> list[float]:
        self.t = 0
        self.step(0)
        return [s.reset(self.influent_flow) for s in self.sensors]

class MixedTimeSeriesInfluent(InfluentModel):
    # Each day, randomly choose a day across all the influent files
    # Preserve the day of the week in this selection to maintain the
    # weekly seasonality
    def __init__(self, sensor_list: tuple["SensorModel", ...] = ()):
        super().__init__(sensor_list)

        data = []
        q_ind = [15]
        x_inds = [*list(range(1, 15)), 16]
        dt_per_day = int(24*60/15) # 15 minute dt
        for file_name in ['dryinfluent', 'raininfluent', 'storminfluent']:
            with resources.open_text(bsm1, "influent_data/" + file_name + '.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                raw_data = np.array(list(reader))
            t = raw_data[:,0]
            # Assume 15 minute dt and daily cycles start at 0
            assert np.allclose(np.diff(t), 1/dt_per_day, atol=1e-6)
            # Divide into separate days
            n_weeks = np.floor_divide(len(raw_data), dt_per_day*7)
            qx_data = raw_data[:n_weeks*7*dt_per_day, q_ind + x_inds]
            data += [np.reshape(qx_data, [n_weeks, 7, dt_per_day, len(q_ind + x_inds)])]

        self.series_t = np.arange(0,dt_per_day)/dt_per_day
        self.series_QX = np.vstack(data)
        self.series_weeks = self.series_QX.shape[0]

        self.t = 0
        self.day = 0
        self.week = np.random.randint(0,self.series_weeks)
        self.t_max = 1. - 1/dt_per_day/2

    def step(self, dt: float) -> list[float]:
        t_ind = np.argmin(np.abs(self.series_t - self.t))
        if self.day % 7 < 5:
            # Weekday, so randomly seelct a weekday from the given week
            day_ind = np.random.randint(0,5)
        else:
            # Weekend, so randomly select a weekend from the given week
            day_ind = np.random.randint(5,7)
        self.influent_flow.X = self.series_QX[self.week, day_ind, t_ind, 1:]
        self.influent_flow.Q = self.series_QX[self.week, day_ind, t_ind, 0]

        self.t += dt
        if self.t > self.t_max:
            self.t -= 1.0
            self.day = (self.day + 1) % 7
            self.week = np.random.randint(0,self.series_weeks)

        return super().step(dt)

    def reset(self) -> list[float]:
        self.t = 0
        self.day = 0
        self.week = np.random.randint(0,self.series_weeks)
        return [s.reset(self.influent_flow) for s in self.sensors]

class SensorModel:
    def __init__(self, measurement: str, sensor_range: list, sensor_class: Literal['A', 'B', 'C', 'D'] = 'A',
                 sensor_noise: float = 0.0):
        self.measurement = measurement
        self.is_Q = (measurement == 'Q')
        self.sensor_range = sensor_range
        self.sensor_class = sensor_class
        self.sensor_noise = 0#sensor_noise
        assert sensor_class in ['A', 'B', 'C', 'D'], "Invalid sensor class"

    def step(self, measured_flow: FlowModel) -> float:
        # Need to step the sensor in order to replicate their dynamic behavior (e.g. time delay)
        if self.is_Q:
            raw_value = measured_flow.Q
        else:
            raw_value = measured_flow.get(self.measurement)
        noisy_value = raw_value + np.random.normal(0, self.sensor_noise)
        # Clip the value to the sensor range
        return np.clip(noisy_value,self.sensor_range[0],self.sensor_range[1])

    def reset(self, measured_flow: FlowModel) -> float:
        # Reset the sensor state if needed (e.g. for dynamic sensors)
        return self.step(measured_flow)

class SbhSensorModel:
    def __init__(self, sensor_range: list, sensor_class: Literal['A', 'B', 'C', 'D'] = 'A', sensor_noise: float = 0.0,
                 solids_threshold: float = 1000.):
        self.sensor_range = sensor_range
        self.sensor_class = sensor_class
        self.sensor_noise = 0#sensor_noise
        self.solids_threshold = solids_threshold

    def step(self, layer_concentrations: np.ndarray, layer_height: float) -> float:
        cum_solids = np.cumsum([*layer_concentrations[::-1].tolist(), self.solids_threshold])
        sbh = np.interp(self.solids_threshold, cum_solids, np.arange(len(layer_concentrations),-1,-1)*layer_height)

        noisy_value = sbh + np.random.normal(0, self.sensor_noise)
        # Clip the value to the sensor range
        return np.clip(noisy_value,self.sensor_range[0],self.sensor_range[1])

    def reset(self, layer_concentrations: np.ndarray, layer_height: float) -> float:
        # Reset the sensor state if needed (e.g. for dynamic sensors)
        return self.step(layer_concentrations, layer_height)
