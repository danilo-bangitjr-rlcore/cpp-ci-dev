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
    inflow: float = 0  # total inflow to the tank
    outflow: float = 0  # total outflow

@dataclass
class Receipt:
    nom: float
    max: float = 1000.

    # state variables
    forecast: float = 0

@dataclass
class Delivery:
    value: float
    max: float = 2000.

@dataclass
class Junction:
    inputs: list[str]
    outputs: list[str]

    # state variables
    flow: float = 0  # net flow through the junction, inflow - out

@dataclass
class Weights:
    volumereward: int
    tankpenalty: int
    deliveryreward: int

@dataclass
class PipelineData:
    segments: dict[str, Segment] = field(default_factory=lambda: {
        't1_t2': Segment(max_flow=2000, min_flow=100, start_cost=100),
        't2_t3': Segment(max_flow=2000, min_flow=100, start_cost=100),
    })
    tanks: dict[str, Tank] = field(default_factory=lambda: {
        't1': Tank(level=25000, capacity=50000),
        't2': Tank(level=25000, capacity=50000),
        't3': Tank(level=25000, capacity=50000),
    })
    receipts: dict[str, Receipt] = field(default_factory=lambda: {
        'r1': Receipt(nom=500),
        'r2': Receipt(nom=500),
    })
    deliveries: dict[str, Delivery] = field(default_factory=lambda: {
        'd1': Delivery(value=1000),
    })
    junctions: dict[str, Junction] = field(default_factory=lambda: {
        'j1': Junction(inputs=['r1'], outputs=['t1']),
        'j2': Junction(inputs=['t1'], outputs=['t1_t2']),
        'j3': Junction(inputs=['t1_t2', 'r2'], outputs=['t2']),
        'j4': Junction(inputs=['t2'], outputs=['t2_t3']),
        'j5': Junction(inputs=['t2_t3'], outputs=['t3']),
        'j6': Junction(inputs=['t3'], outputs=['d1']),
    })
    weights: Weights = field(default_factory=lambda: Weights(volumereward=100, tankpenalty=1000, deliveryreward=10))


@dataclass
class PipelineConfig(EnvConfig):
    name: str = 'Pipeline-v0'
    pipeline_data: PipelineData = field(default_factory=PipelineData)

class PipelineEnv(gym.Env):
    def __init__(self, cfg: PipelineConfig):
        self.segments = cfg.pipeline_data.segments
        self.tanks = cfg.pipeline_data.tanks
        self.receipts = cfg.pipeline_data.receipts
        self.deliveries = cfg.pipeline_data.deliveries
        self.junctions = cfg.pipeline_data.junctions
        self.weights = cfg.pipeline_data.weights
        self.nodes = (
            list(self.segments.keys())
            + list(self.tanks.keys())
            + list(self.receipts.keys())
            + list(self.deliveries.keys())
        )
        # Action space is the flow rates on each pipeline segment
        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.segments),), dtype=np.float64)
        # Observation space includes tank levels, forecasted receipts, and incurred stops and starts on the segments
        n_obs = len(self._get_obs())
        self.observation_space = gym.spaces.Box(np.zeros(n_obs),np.ones(n_obs))
        self.observation = self.observation_space.sample()
        self.reward = 0
        self.t = 0

    def reset(self, seed: int | None = None, options: dict | None = None): # type: ignore
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # for k in self.tanks.keys():
        #     self.tanks[k].level = self.np_random.random() * self.tanks[k].capacity

        for s in self.segments.values():
            s.flowing = 0
            s.start = 0
            s.stop = 0
        for r in self.receipts.values():
            r.forecast = r.nom
        self.reward = 0
        self.t = 0

        # Reset Tank levels, etc.
        return self._get_obs(), {}


    def _get_obs(self):
        return np.hstack([np.array([tank.level / tank.capacity for tank in self.tanks.values()],dtype='float32'),
                          np.array([recpt.forecast/recpt.max for recpt in self.receipts.values()],dtype='float32'),
                          np.array([deliv.value for deliv in self.deliveries.values()],dtype='float32'),
                          np.array([seg.stop for seg in self.segments.values()],dtype='float32'),
                          np.array([seg.start for seg in self.segments.values()],dtype='float32')])

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

        for key in self.junctions:
            j = self.junctions[key]
            if any([t.level<=0 for key, t in self.tanks.items() if key in j.inputs()]) or \
               any([t.level>=t.capacity for key, t in self.tanks.items() if key in j.inputs()]):
                for s in self.segments:
                    if s in j.inputs:
                        self.segments[s].flow = 0.
                        self.segments[s].flowing = 0.
                        self.segments[s].stop = 1
                        self.segments[s].start = 0
                j.flow = 0.   
            else:
                inflow = sum([s.flow for key, s in self.segments.items() if key in j.inputs])
                inflow += sum([r.forecast for key, r in self.receipts.items() if key in j.inputs])
                outflow = sum([s.flow for key, s in self.segments.items() if key in j.outputs])
                outflow += sum([r.value for key, r in self.deliveries.items() if key in j.outputs])
                j.flow = inflow + outflow

        for tank_name, t in self.tanks.items():
            t.inflow = sum([j.flow for j in self.junctions.values() if tank_name in j.outputs])
            t.outflow = sum([j.flow for j in self.junctions.values() if tank_name in j.inputs])
            t.level += t.inflow - t.outflow

            perc = t.level / t.capacity
            volumeaward = (1-perc)*(perc)*self.weights.volumereward
            self.reward += volumeaward
            if (perc > 1) or (perc < 0):
                self.reward += -self.weights.tankpenalty
                # truncated = True


        # Calculate rewards/penalties
        for d in self.deliveries.values():
            self.reward += d.value*self.weights.deliveryreward
        self.t +=1
        observation = self._get_obs()
        reward = self.reward
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
        'd1': Delivery(value=1000),
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
