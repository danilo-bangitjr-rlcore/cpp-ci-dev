from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np

from rl_env.factory import EnvConfig, env_group


@dataclass
class BSM1Config(EnvConfig):
    name: str = 'BSM1-v0'

class BSM1Env(gym.Env):
    """
    Environment based on the BSM1 benchmark model for wastewater treatment plants:
    http://iwa-mia.org/wp-content/uploads/2019/04/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf

    Some of this code is based off the implementation here:
    https://gitlab.rrze.fau.de/evt/klaeffizient/asm-python/-/tree/main

                                               IMLR
           ┌───────────────────────────────────────────────────────────────────┐
           │                                                                   │
           │                                                                   │
      ┌────▼─────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    ┌─────┴────┐      xxxxxxxxxxxxxxx
      │          │     │          │     │          │     │          │    │          │       x clarifier x
    ──►  Tank 1  ├─────►  Tank 2  ├─────►  Tank 3  ├─────►  Tank 4  ├────►  Tank 5  ├────────►─────────x
      │          │     │          │     │    ▲     │     │    ▲     │    │     ▲    │         x───────x
      └────▲─────┘     └──────────┘     └────┼─────┘     └────┼─────┘    └─────┼────┘          x─────x
           │                                 │                │                │                x───x
           │                                 │                │                │                 xxx
           │           Aeration   ───────────┴────────────────┴────────────────┘                  │
           │                                                                                      │
           └──────────────────────────────────────────────────────────────────────────────────────┤
                                                RAS                                               │
                                                                                                  └────►
                                                                                                    WAS

    Each tank is modeled using the ASM1 activated sludge model.
    The state of the system is the biochemical composition of each tank + each layer of the clarifier.
    The actions are given by aeration to each of tanks 3,4,5, the recycle rates (IMLR and RAS), and the
    wasteage rate (WAS)
    """
    def __init__(self, cfg:  BSM1Config):
        self._params = BSM1Params()

        self._influent = InfluentModel(self._params.influent_sensors)
        self._tanks = [TankModel(self._params.tank_sensors[i]) for i in range(5)]
        self._clarifier = ClarifierModel(self._params.effluent_sensors)

        action_range = np.vstack([np.tile(self._params.aeration_range,[3,1]),
                                  self._params.ILMR_range,
                                  self._params.RAS_range,
                                  self._params.WAS_range])
        self.action_space = gym.spaces.Box(action_range[:,0], action_range[:,1], dtype=np.float64)

        sensor_range = [s.sensor_range for s in self._influent.sensors]
        for t in self._tanks:
            sensor_range += [s.sensor_range for s in t.sensors]
        sensor_range += [s.sensor_range for s in self._clarifier.sensors]
        sensor_range = np.vstack(sensor_range)
        self.observation_space = gym.spaces.Box(sensor_range[:,0], sensor_range[:,1], dtype=np.float64)

        return

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        aeration = np.zeros(5)
        aeration[2:] = action[:3]
        Q_IMLR = action[3]
        Q_RAS = action[4]
        Q_WAS = action[5]

        obs = []

        obs += self._influent.step()
        IMLR_flow = FlowModel(self._tanks[-1].tank_flow.X, Q_IMLR)
        RAS_flow = FlowModel(self._clarifier.wasteage_flow.X, Q_RAS)

        # Flow through the 5 tanks
        #   Add influent + RAS + IMLR to get first tank flow
        obs += self._tanks[0].step(self._influent.influent_flow + IMLR_flow + RAS_flow, aeration[0])
        for i in range(1,5):
            obs += self._tanks[i].step(self._tanks[i-1].tank_flow, aeration[i])
        
        # Flow through the clarifier
        obs += self._clarifier.step(self._tanks[-1].tank_flow, Q_RAS + Q_WAS)

        return np.array(obs), 0.0, False, False, {} 
    
    def reset(self, *, seed = None, options = None):
        self._influent = InfluentModel(self._params.influent_sensors)
        self._tanks = [TankModel(self._params.tank_sensors[i]) for i in range(5)]
        self._clarifier = ClarifierModel(self._params.effluent_sensors)

        obs,_,_,_,_ = self.step(self.action_space.sample())

        return np.array(obs), {}

class FlowModel:
    state_names = ['SI',   'SS',  'XI',  'XS',  'XBH', 'XBA',  'XP',
                   'SO',   'SNO', 'SNH', 'SND', 'XND', 'SALK', 'TSS',
                   'TEMP', 'SD1', 'SD2', 'SD3', 'XD4', 'XD5']
    state_idx = {n: i for i, n in enumerate(state_names)}
    nX = len(state_names)

    def __init__(self, X: np.ndarray = None, Q: float = 0):
        if X is None:
            X = np.zeros(self.nX)
        assert len(X) == self.nX, 'Incorrect state array size'

        self.X = X # Flow state: concentrations, temperature, pH, etc
        self.Q = Q # Flow rate
        pass

    def __add__(self, other:"FlowModel")->"FlowModel":
        combined_Q = self.Q + other.Q
        combined_X = (self.X*self.Q + other.X*other.Q)/combined_Q
        return FlowModel(combined_X, combined_Q)

    def get(self, name: 'str')->float:
        return self.X[self.state_idx[name]]

    def copy(self)->"FlowModel":
        return FlowModel(self.X.copy(),self.Q)

    def copy_from(self, other:"FlowModel"):
        self.X[:] = other.X
        self.Q = other.Q

class TankModel:
    """
    Based on ASM1
    """
    def __init__(self, sensor_list: list["SensorModel"]):
        self.tank_flow = FlowModel()
        self.sensors = sensor_list

    def step(self, influent: FlowModel, aeration: float):
        self.tank_flow.copy_from(influent)

        obs = [s.step(self.tank_flow) for s in self.sensors]
        return obs

class ClarifierModel:
    def __init__(self, sensor_list: list["SensorModel"]):
        self.effluent_flow = FlowModel()
        self.wasteage_flow = FlowModel()
        self.sensors = sensor_list

    def step(self, influent: FlowModel, sludge_flow_rate: float):
        self.effluent_flow.copy_from(influent)
        self.wasteage_flow.copy_from(influent)

        obs = [s.step(self.effluent_flow) for s in self.sensors]
        return obs

class InfluentModel:
    def __init__(self, sensor_list: list["SensorModel"]):
        self.influent_flow = FlowModel(np.random.random([20]), 1)
        self.sensors = sensor_list

    def step(self):
        obs = [s.step(self.influent_flow) for s in self.sensors]
        return obs

class SensorModel:
    def __init__(self, measurement: str, sensor_range: list):
        self.measurement = measurement
        self.sensor_range = sensor_range

    def step(self, measured_flow: FlowModel) -> tuple[FlowModel, list]:
        # Need to step the sensor in order to replicate their dynamic behavior (e.g. time delay)
        raw_value = measured_flow.get(self.measurement)
        clipped_value = np.clip(raw_value,self.sensor_range[0],self.sensor_range[1])
        return clipped_value

@dataclass
class BSM1Params():
    # Action ranges
    aeration_range: ClassVar[list] = [0,1]
    ILMR_range: ClassVar[list] = [0,1]
    RAS_range: ClassVar[list] = [0,1]
    WAS_range: ClassVar[list] = [0,1]
    influent_sensors:  ClassVar[list[SensorModel]]  =  [SensorModel('SNH', [0,1]), SensorModel('SO', [0,1])]
    tank_sensors: ClassVar[list[list[SensorModel]]] = [[SensorModel('SNH', [0,1])],
                                                       [],
                                                       [],
                                                       [],
                                                       [SensorModel('SO',  [0,1])]]
    effluent_sensors:  ClassVar[list[SensorModel]]  =  [SensorModel('SNH', [0,1]), SensorModel('SNO', [0,1])]

env_group.dispatcher(BSM1Config(),BSM1Env)