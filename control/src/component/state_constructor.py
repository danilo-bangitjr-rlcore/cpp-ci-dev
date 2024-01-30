
import numpy as np
from copy import deepcopy
from normalizer import Scale
import math
from gymnasium.spaces.utils import flatdim


class BaseStateConstructor:
    def __init__(self):
        self.parents = []
        self.children = None
        self.called = False
        self.o_next = None
        return
    
    def __call__(self, o):
        if not self.called:
            if len(self.parents) == 0: # base case
                o_parents = [o]
            else:
                o_parents = [p(o) for p in self.parents]
            self.o_next = self.process_observation(o_parents)
            self.called = True
        return self.o_next
    
    
    def set_parents(self, parents):
        self.parents = parents
        
        
    def set_children(self, children):
        self.children = children
        
        
    def reset_called(self):
        self.called = False
        for p in self.parents:
            p.reset_called()
            
        
    def process_observation(self, o_parents):
        """
        takes a list and returns a VECTOR
        """
        raise NotImplementedError
    
    def _clear_state(self):
        return
    
    def clear_state(self):
        self._clear_state()
        for p in self.parents:
            p.clear_state()
        


class Identity(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert len(o_parents) == 1
        return o_parents[0]



class KOrderHistory(BaseStateConstructor):
    """
    Keeps a running list of observations
    """
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.obs_history = []
        self.num_elements = 0
        
        
    def process_observation(self, o_parents):
        """
        takes a list and returns a VECTOR
        """
        assert(len(o_parents)) == 1
        o = o_parents[0]
        self.obs_history.append(o)
        
        if len(self.obs_history) > self.k:
            self.obs_history.pop(0)
             
        return_list = deepcopy(self.obs_history)
        # ensure returned list has self.k elements
        if len(return_list) < self.k:
            last_element = return_list[-1]
            for _ in range(len(return_list), self.k):
                return_list.append(last_element)
                
        return np.array(return_list)
    
    def _clear_state(self):
        self.obs_history = []
        self.num_elements = 0
    


class MemoryTrace(BaseStateConstructor):
    def __init__(self, trace_decay):
        super().__init__()
        self.trace_decay = trace_decay
        self.trace = None

    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0]
        if self.trace is None: # first observation received
            self.trace = (1-self.trace_decay)*o + self.trace_decay*np.zeros_like(o)
        else:
            self.trace = (1-self.trace_decay)*o + self.trace_decay*self.trace
        return self.trace
    
    def _clear_state(self):
        self.trace = None


class IntraStepMemoryTrace(BaseStateConstructor):
    def __init__(self, trace_decay):
        super().__init__()
        self.trace_decay = trace_decay
        self.trace = None

    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0]
        num_rows = o.shape[0]
        for i in range(num_rows):
            row = o[i, :]
            if self.trace is None: # first observation received
                self.trace = (1-self.trace_decay)*row + np.zeros_like(row)
            else:
                self.trace = (1-self.trace_decay)*row + self.trace_decay*self.trace
        return self.trace
    
    def _clear_state(self):
        self.trace = None


class SubSample(BaseStateConstructor):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0].copy()
        
        if len(o.shape) == 1:
            return o
        
        elif len(o.shape) == 2:
            o = np.flip(o, axis=0)
            o = o[0::self.n]
            return np.flip(o, axis=0)
        else:
            raise NotImplementedError


class LastK(BaseStateConstructor):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o_parent = o_parents[0].copy()
        if len(o_parent.shape) == 2: # 2D array
            o = o_parent[-self.k:, ]
        elif len(o_parent.shape) == 1:
            o = o_parent[-self.k:]    
        return o


class Flatten(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        return o_parents[0].flatten()


class Concatenate(BaseStateConstructor):
    def process_observation(self, o_parents):
        return np.concatenate(o_parents, axis=0)


class Normalize(BaseStateConstructor):
    def __init__(self, scaler, bias):
        super().__init__()
        self.normalizer = Scale(scaler, bias)
        
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o_parent = o_parents[0]
        return self.normalizer(o_parent)


class ReseauNormalize(Normalize):
    def __init__(self):
        super().__init__()
        scaler = 1
        bias = 1
        self.normalizer = Scale(scaler, bias)
        


class WindowAverage(BaseStateConstructor):
    """
    Averages every window_size observations together
    """
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
    
    
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0].copy()
        assert(len(o.shape)==2)
        num_rows =  o.shape[0] 
        o = o[num_rows%self.window_size:]
        assert o.shape[0] % self.window_size == 0
        o = o.reshape(-1, self.window_size, o.shape[1])
        return np.mean(o, axis = 1)



class Beginning(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0].copy()
        assert(len(o.shape)==2)
        return o[0, :]
    
    
class End(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0].copy()
        assert(len(o.shape)==2)
        return o[-1, :]
    
    
class Mid(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0].copy()
        assert(len(o.shape)==2)
        i = math.ceil(o.shape[0]/2)
        return o[i, :]


class StateConstructorWrapper:
    def __init__(self, state_constructor, time_frame=1):
        self.state_constructor = state_constructor
        self.time_frame = time_frame
        
    def __call__(self, o):
        state = self.state_constructor(o)
        self.state_constructor.reset_called()
        return state
    
    def get_state_dim(self, dim):
        # dim = flatdim(env.observation_space)
        if self.time_frame == 1:
            test_obs = np.ones(dim)
        else:
            test_obs = np.ones((self.time_frame, dim))
        state = self(test_obs)
        assert len(state.shape) == 1
        state_dim = state.shape[0]
        self.state_constructor.clear_state()
        return state_dim
        
        

def init_state_constructor(name, cfg):
    if name == "Identity":
        s1 = Identity()
        sc = StateConstructorWrapper(s1)
        return sc
    
    elif name == "Reseau_order_k":
        s1 = ReseauNormalize()
        s2 = WindowAverage(cfg.window_average)
        s2.set_parents([s1])

        s3 = End()
        s3.set_parents([s2])

        s4 = KOrderHistory(cfg.k_order_hist)
        s4.set_parents([s3])

        s5 = Flatten()
        s5.set_parents([s4])
        sc = StateConstructorWrapper(s5)
        return sc
    
    elif name == "Reseau_single_trace":
        # Single trace
        s1 = ReseauNormalize()
        s2 = WindowAverage(cfg.window_average)
        s2.set_parents([s1])
        s3 = End()
        s3.set_parents([s2])
        s4 = MemoryTrace(cfg.trace_decay)
        s4.set_parents([s3])
        s5 = Concatenate()
        s5.set_parents([s3, s4])
        sc = StateConstructorWrapper(s5)
        return sc
        
    elif name == "Reseau_double_trace":
        # Double trace
        s1 = ReseauNormalize()
        s2 = WindowAverage(cfg.window_average)
        s2.set_parents([s1])

        s3 = End()
        s3.set_parents([s2])

        s4 = MemoryTrace(cfg.trace_decay)
        s4.set_parents([s3])

        s5 = SubSample(2)
        s5.set_parents([s2])

        s6 = IntraStepMemoryTrace(cfg.intra_step_trace_decay)
        s6.set_parents([s5])

        s7 = Concatenate()

        s7.set_parents([s3, s4, s6])
        sc = StateConstructorWrapper(s7)
        return sc

    else:
        raise NotImplementedError