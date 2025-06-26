from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rl_env.factory import EnvConfig, env_group


@dataclass
class Segment:
    max_flow: float
    min_flow: float
    start_cost: int

    # state variables
    flowing: int = 0  # 1 if the segment is flowing, 0 otherwise
    flow: float = 0  # current flow rate
    start: int = 0  # 1 if the segment was started this step, 0 otherwise
    stop: int = 0  # 1 if the segment was stopped

@dataclass
class Tank:
    level: float
    capacity: float

    # state variables
    start_volume: float = 0  # volume at the start of the step
    inflow: float = 0  # total inflow to the tank
    outflow: float = 0  # total outflow
    end_volume: float = 0  # volume at the end of the step, calculated

@dataclass
class Receipt:
    nom: float

    # state variables
    forecast: float = 0

@dataclass
class Delivery:
    values: list[float]

@dataclass
class Junction:
    inputs: list[str]
    outputs: list[str]

    # state variables
    inflow: float = 0  # total inflow to the junction
    outflow: float = 0  # total outflow from the junction
    flow: float = 0  # net flow through the junction, inflow - out

@dataclass
class Weights:
    volumereward: int
    tankpenalty: int
    deliveryreward: int

@dataclass
class PipelineData:
    segments: dict[str, Segment] = field(default_factory=dict)
    tanks: dict[str, Tank] = field(default_factory=dict)
    receipts: dict[str, Receipt] = field(default_factory=dict)
    deliveries: dict[str, Delivery] = field(default_factory=dict)
    junctions: dict[str, Junction] = field(default_factory=dict)
    weights: Weights = field(default_factory=lambda: Weights(volumereward=100, tankpenalty=1000, deliveryreward=10))


@dataclass
class PipelineConfig(EnvConfig):
    name: str = 'Pipeline-v0'
    pipeline_data: PipelineData = field(default_factory=PipelineData)

class PipelineEnv(gym.Env):
    def __init__(self, cfg: PipelineConfig):
        self.horizon = cfg.pipeline_data.horizon
        self.segments = cfg.pipeline_data.segments
        self.tanks = cfg.pipeline_data.tanks
        self.receipts = cfg.pipeline_data.receipts
        self.deliveries = cfg.pipeline_data.deliveries
        self.junctions = cfg.pipeline_data.junctions
        self.weights = cfg.pipeline_data.weights
        self.nodes = list(self.segments.keys()) + list(self.tanks.keys()) + list(self.receipts.keys()) + list(self.deliveries.keys())
        # Action space is the flow rates on each pipeline segment
        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.segments),), dtype=np.float64)
        # Observation space includes tank levels, forecasted receipts, and incurred stops and starts on the segments
        self.observation_space = gym.spaces.Dict(
            {
                "tank_levels": gym.spaces.Box(0, 1, shape=(len(self.tanks),), dtype=np.float64),
                "receipts_forecast": gym.spaces.Box(0, 1, dtype=np.float64),
                "stops": gym.spaces.MultiBinary(len(self.segments)),
                "starts": gym.spaces.MultiBinary(len(self.segments)),
            },
        )
        self.observation = self.observation_space.sample()
        self.reward = 0
        self.t = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.observation = self.observation_space.sample()
        for i, k in enumerate(self.tanks):
            self.tanks[k].level = self.observation['tank_levels'][i]
        for k, s in self.segments.items():
            s.flowing = 0
            s.start = 0
            s.stop = 0
        for k, r in self.receipts.items():
            r.forecast = r.nom * 2 in self.observation['receipts_forecast']
        self.reward = 0
        self.t = 0

        # Reset Tank levels, etc.
        return self._get_obs(), {}


    def _get_obs(self):
        return {
                "tank_levels": np.array([item.level for key, item in self.tanks.items()]),
                "receipts_forecast": self.observation['receipts_forecast'],
                "stops": np.array([s.stop for i, s in self.segments.items()]),
                "starts": np.array([s.start for i, s in self.segments.items()]),
            }


    def step(self, action: np.ndarray):
        terminated = False
        truncated = False
        for i, key in enumerate(self.segments):
            s = self.segments[key]
            flowing = action[i] > 0
            flow = action[i]*s.max_flow
            minflow = flow > s.min_flow

            if flowing and not s.flowing:
                s.start = 1
                self.reward -= s.start_cost
            else:
                s.start = 0
            if not minflow:
                s.stop = 1
            else:
                s.stop = 0
            s.flowing = minflow
            s.flow = flow

        for i, key in enumerate(self.junctions):
            j = self.junctions[key]
            j.inflow = sum([s.flow for key, s in self.segments.items() if key in j.inputs])
            j.inflow += sum([r.forecast for key, r in self.receipts.items() if key in j.inputs])
            j.outflow = sum([s.flow for key, s in self.segments.items() if key in j.outputs])
            j.outflow += sum([r.values[0] for key, r in self.deliveries.items() if key in j.outputs])
            j.flow = j.inflow+ j.outflow

        for i, t in self.tanks.items():
            t.start_volume = t.level * t.capacity
            t.inflow = sum([j.flow for k, j in self.junctions.items() if i in j.outputs])
            t.outflow = sum([j.flow for k, j in self.junctions.items() if i in j.inputs])
            t.end_volume = t.start_volume  + t.inflow - t.outflow
            t.level = t.end_volume / t.capacity
            volumeaward = (1-t.level)*(t.level)*self.weights.volumereward
            self.reward += volumeaward
            if (t.level > 1) or (t.level < 0):
                self.reward += -self.weights.tankpenalty
                truncated = True


        # Calculate rewards/penalties
        for i, d in self.deliveries.items():
            self.reward += d.values[0]*self.weights.deliveryreward
        self.t +=1
        observation = self._get_obs()
        reward = self.reward
        if self.t >= self.horizon -1:
            terminated = True
        info = {"segments": self.segments, "tanks": self.tanks}
        return observation, reward, terminated, truncated, info

env_group.dispatcher(PipelineConfig(), PipelineEnv)

dm = PipelineData(
    segments={
        't1_t2': Segment(max_flow=2000, min_flow=100, start_cost=100),
        't2_t3': Segment(max_flow=2000, min_flow=100, start_cost=100),
    },
    tanks={
        't1': Tank(level=500, capacity=5000),
        't2': Tank(level=500, capacity=5000),
        't3': Tank(level=500, capacity=5000),
    },
    receipts={
        'r1': Receipt(nom=500),
        'r2': Receipt(nom=500),
    },
    deliveries={
        'd1': Delivery(values=[1000]),
    },
    junctions={
        'j1': Junction(inputs=['r1'], outputs=['t1']),
        'j2': Junction(inputs=['t1'], outputs=['t1_t2']),
        'j3': Junction(inputs=['t1_t2', 'r2'], outputs=['t2']),
        'j4': Junction(inputs=['t2'], outputs=['t2_t3']),
        'j5': Junction(inputs=['t2_t3'], outputs=['t3']),
        'j6': Junction(inputs=['t3'], outputs=['d1']),
    },
    weights=Weights(
        volumereward=100,
        tankpenalty=1000,
        deliveryreward=10,
    ),
)
