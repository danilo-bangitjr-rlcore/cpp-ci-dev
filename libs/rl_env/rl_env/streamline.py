from typing import Optional
import numpy as np
import gymnasium as gym

class PipelineEnv(gym.Env):
    def __init__(self, dm):
        self.horizon = dm['horizon']
        self.segments = dm['segments']
        self.tanks = dm['tanks']
        self.receipts = dm['receipts']
        self.deliveries = dm['deliveries']
        self.junctions = dm['junctions']
        self.weights = dm['weights']
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
            }
        )
        self.observation = self.observation_space.sample()
        self.reward = 0
        self.t = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.observation = self.observation_space.sample()
        for i, k in enumerate(self.tanks):
            self.tanks[k]['level'] = self.observation['tank_levels'][i]
        for k, s in self.segments.items():
            s['flowing'] = 0
            s['start'] = 0
            s['stop'] = 0
        for k, r in self.receipts.items():
            r['forecast'] = r['nom'] * 2 in self.observation['receipts_forecast']
        self.reward = 0
        self.t = 0

        # Reset Tank levels, etc.
        return self._get_obs(), {}


    def _get_obs(self):
        return {
                "tank_levels": np.array([item['level'] for key, item in self.tanks.items()]),
                "receipts_forecast": self.observation['receipts_forecast'],
                "stops": np.array([s['stop'] for i, s in self.segments.items()]),
                "starts": np.array([s['start'] for i, s in self.segments.items()])
            }
    
   
    def step(self, action):
        terminated = False
        truncated = False
        for i, key in enumerate(self.segments):
            s = self.segments[key]
            flowing = action[i] > 0
            flow = action[i]*s['max_flow']
            minflow = flow > s['min_flow']
            
            if flowing and not s['flowing']:
                s['start'] = 1
                self.reward -= s['start_cost']
            else:
                s['start'] = 0
            if not minflow:
                s['stop'] = 1
            else:
                s['stop'] = 0
            s['flowing'] = minflow
            s['flow'] = flow

        for i, key in enumerate(self.junctions):
            j = self.junctions[key]
            j['inflow'] = sum([s['flow'] for key, s in self.segments.items() if key in j['inputs']])
            j['inflow'] += sum([r['forecast'] for key, r in self.receipts.items() if key in j['inputs']])
            j['outflow'] = sum([s['flow'] for key, s in self.segments.items() if key in j['outputs']])
            j['outflow'] += sum([r[0] for key, r in self.deliveries.items() if key in j['outputs']])
            j['flow'] = j['inflow']+ j['outflow']

        for i, t in self.tanks.items():
            t['start_volume']= t['level'] * t['capacity']
            t['inflow'] = sum([j['flow'] for k, j in self.junctions.items() if i in j['outputs']])
            t['outflow'] = sum([j['flow'] for k, j in self.junctions.items() if i in j['inputs']])
            t['end_volume'] = t['start_volume'] + t['inflow'] - t['outflow']
            t['level'] = t['end_volume'] / t['capacity']
            volumeaward = (1-t['level'])*(t['level'])*self.weights['volumereward']
            self.reward += volumeaward
            if (t['level'] > 1) or (t['level'] < 0):
                self.reward += -self.weights['tankpenalty']
                truncated = True

            
        # Calculate rewards/penalties
        for i, d in self.deliveries.items():
            self.reward += d[0]*self.weights['deliveryreward']
        self.t +=1
        observation = self._get_obs()
        reward = self.reward
        if self.t >= self.horizon -1:
            terminated = True
        info = {"segments": self.segments, "tanks": self.tanks}
        return observation, reward, terminated, truncated, info


dm = {
    "horizon": 10,
    "segments": {
            't1_t2':{'max_flow':2000, 'min_flow':100, 'start_cost':100},
            't2_t3':{'max_flow':2000, 'min_flow':100, 'start_cost':100}
            },
    "tanks": {
            't1':{'level':500, 'capacity': 5000},
            't2':{'level':500, 'capacity': 5000},
            't3':{'level':500, 'capacity': 5000}
        },
    "receipts": {
            'r1':{'nom':500},
            'r2':{'nom':500}
            },
    "deliveries": {
            'd1':[1000]
            },
    "junctions": {
            'j1':{'inputs':['r1'], 'outputs':['t1']},
            'j2':{'inputs':['t1'], 'outputs':['t1_t2']},
            'j3':{'inputs':['t1_t2', 'r2'], 'outputs':['t2']},
            'j4':{'inputs':['t2'], 'outputs':['t2_t3']},
            'j5':{'inputs':['t2_t3'], 'outputs':['t3']},
            'j6':{'inputs':['t3'], 'outputs':['d1']}
            },
     "weights": {
                'volumereward':100,
                'tankpenalty':1000,
                'deliveryreward':10
                
     }
    }

