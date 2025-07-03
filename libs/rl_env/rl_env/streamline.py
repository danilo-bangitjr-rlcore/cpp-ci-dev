from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import numpy as np
from scipy.optimize import linprog

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
        't1': Tank(level=2500, capacity=5000),
        't2': Tank(level=2500, capacity=5000),
        't3': Tank(level=2500, capacity=5000),
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
    control_strategy: Literal['agent', "mpc", "dagger"] = "dagger"

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
        self.control_strategy = cfg.control_strategy

    def reset(self, seed: int | None = None, options: dict | None = None): # type: ignore
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        for k in self.tanks.keys():
            self.tanks[k].level = self.np_random.random() * self.tanks[k].capacity

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
        return np.hstack(
            [np.array([tank.level / tank.capacity for tank in self.tanks.values()],dtype='float32'),
             np.array([recpt.forecast / recpt.max for recpt in self.receipts.values()],dtype='float32'),
             np.array([deliv.value / deliv.max for deliv in self.deliveries.values()],dtype='float32'),
             np.array([seg.stop for seg in self.segments.values()],dtype='float32'),
             np.array([seg.start for seg in self.segments.values()],dtype='float32')],
        )

    def step(self, action: np.ndarray):
        terminated = False
        truncated = False
        use_mpc = False

        if self.control_strategy == 'mpc':
            use_mpc = True
        elif self.control_strategy == 'dagger':
            mpc_ratio = np.clip(1-0.003*self.t,0,1)
            use_mpc = np.random.rand() < mpc_ratio
        if use_mpc:
            action = self._solve_mpc(np.array([tank.level / tank.capacity for tank in self.tanks.values()]),
                                     np.array([recpt.forecast for recpt in self.receipts.values()]),
                                     np.array([deliv.value for deliv in self.deliveries.values()]))

        for i, key in enumerate(self.segments):
            s = self.segments[key]
            flowing = action[i] > 0
            flow = action[i]*s.max_flow

            for j in self.junctions.values():
                if key in j.outputs:
                    for t_key, t in self.tanks.items():
                        if t_key in j.inputs and t.level <= 0.01:
                            flow = 0
                            flowing = False

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
            if (perc >= 1) or (perc <= 0):
                self.reward += -self.weights.tankpenalty

                t.level = np.clip(t.level,0,t.capacity)
                # truncated = True


        # Calculate rewards/penalties
        for d in self.deliveries.values():
            self.reward += d.value*self.weights.deliveryreward

        # Update receipts and deliveries
        if np.mod(self.t,10) == 0:
            for d in self.deliveries:
                self.deliveries[d].value = np.random.random()*self.deliveries[d].max
            for r in self.receipts:
                self.receipts[r].forecast = np.random.random()*self.receipts[r].max
            total_deliveries = np.sum([self.deliveries[d].value for d in self.deliveries])
            total_receipts = np.sum([self.receipts[r].forecast for r in self.receipts])
            for d in self.deliveries:
                self.deliveries[d].value = self.deliveries[d].value*total_receipts/total_deliveries

        self.t +=1
        observation = self._get_obs()
        reward = self.reward
        info = {"action_override": action, "segments": self.segments, "tanks": self.tanks}
        return observation, reward, terminated, truncated, info

    def _solve_mpc(self, x_0, receipt_forecast, delivery):
        # Variables: [x_1,...,x_N,u_0,...,u_N-1,lambda_0,lambda_1]
        n_steps = 10
        final_soft_bounds = np.array([[0.4,0.6]]*3)
        min_flow = np.array([self.segments[s].min_flow for s in self.segments])
        max_flow = np.array([self.segments[s].max_flow for s in self.segments])
        capacity = np.array([tank.capacity for tank in self.tanks.values()])
        u_lb = min_flow/max_flow+1e-3
        bounds = [(0,1)]*3*n_steps + [(lb,1) for lb in u_lb]*n_steps + [(0,None)]*3

        _A = np.eye(3)
        _B = np.vstack([np.array([-1/capacity[0],1/capacity[1],0])*max_flow[0],
                        np.array([0,-1/capacity[1],1/capacity[2]])*max_flow[1]]).T
        _C = np.hstack([receipt_forecast, -delivery])/capacity

        A_eq, b_eq, A_ub, b_ub, c = self._lp_mats(_A, _B, _C, final_soft_bounds, n_steps, x_0)

        sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if sol.x is not None:
            x_out = sol.x
            n_states = len(_A)
            n_actions = np.size(_B,1)
            # Grab the first action out of the solution
            mpc_actions = x_out[n_states*n_steps : n_states*n_steps + n_actions]
        else:
            mpc_actions = np.ones(n_actions)*0.5

        return mpc_actions

    def _lp_mats(self, A, B, C, soft_bounds, n_steps, x_0):
        n_states = len(A)
        n_actions = np.size(B,1)

        A_mat = np.zeros([n_states*n_steps,(n_states+n_actions)*n_steps + 3])
        for i in range(n_steps):
            if i > 0:
                A_mat[i*n_states : (i+1)*n_states, (i-1)*n_states : i*n_states] = A[:,:]
            A_mat[i*n_states : (i+1)*n_states, n_steps*n_states+i*n_actions : n_steps*n_states+(i+1)*n_actions] = B[:,:]

        A_eq = np.hstack([np.eye(n_states*n_steps),np.zeros([n_states*n_steps,n_actions*n_steps+3])]) - A_mat
        b_eq = (np.hstack([x_0,np.zeros(n_states*(n_steps-1))]) + np.tile(C,[1,n_steps])).T

        # Soft state bounds
        # -x_i - lambda_0 <= -soft_min
        #  x_i - lambda_1 <=  soft_max
        #  u_i - lambda_2 <= 0
        # Minimize lambda
        A_ub = np.zeros([2*n_states*n_steps+n_actions*n_steps,(n_states+n_actions)*n_steps+3])
        A_ub[:n_states*n_steps,                   :n_states*n_steps] = -np.eye(n_states*n_steps)
        A_ub[n_states*n_steps:2*n_states*n_steps, :n_states*n_steps] =  np.eye(n_states*n_steps)
        A_ub[:n_states*n_steps,                   (n_states+n_actions)*n_steps  ] = -1
        A_ub[n_states*n_steps:2*n_states*n_steps, (n_states+n_actions)*n_steps+1] = -1
        A_ub[2*n_states*n_steps:, n_states*n_steps:(n_states+n_actions)*n_steps] = np.eye(n_actions*n_steps)
        A_ub[2*n_states*n_steps:, (n_states+n_actions)*n_steps+2] = -1


        b_ub = np.hstack([-soft_bounds[:,0]]*n_steps + [soft_bounds[:,1]]*n_steps + [np.zeros(n_actions*n_steps)])

        # Minimize lambda and u
        c = np.hstack([np.zeros(n_states*n_steps),np.ones(n_actions*n_steps),1000*np.ones(2),100])

        return A_eq, b_eq, A_ub, b_ub, c


env_group.dispatcher(PipelineConfig(), PipelineEnv)
