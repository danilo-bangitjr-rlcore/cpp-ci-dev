
import numpy as np
from copy import deepcopy
from normalizer import Scale
import math


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
        o_parent = o_parents[0]
        self.obs_history.append(o_parent)
        
        if len(self.obs_history) > self.k:
            self.obs_history.pop(0)
             
        return_list = deepcopy(self.obs_history)
        # ensure returned list has self.k elements
        if len(return_list) < self.k:
            last_element = return_list[-1]
            for _ in range(len(return_list), self.k):
                return_list.append(last_element)
                
        return np.array(return_list)
    


class MemoryTrace(BaseStateConstructor):
    def __init__(self, trace_decay):
        super().__init__()
        self.trace_decay = trace_decay
        self.trace = None

    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o_parent = o_parents[0]
        if self.trace is None: # first observation received
            self.trace = o_parent
        else:
            self.trace = (1-self.trace_decay)*o_parent + self.trace_decay*self.trace
        return self.trace


class LastK(BaseStateConstructor):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.trace = None

    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o_parent = o_parents[0]
        if len(o_parent.shape) == 2: # 2D array
            o = o_parent[-self.k:, ]
            o = o.flatten()
        elif len(o_parent.shape) == 1:
            o = o_parent[-self.k:]    
        return o


class Flatten:
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


class WindowAverage(BaseStateConstructor):
    """
    Averages every window_size observations together
    """
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
    
    
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0]
        assert(len(o.shape)==2)
        num_rows =  o.shape[0] 
        o = o[num_rows%self.window_size:]
        assert o.shape[0] % self.window_size == 0
        o = o.reshape(-1, self.window_size, o.shape[1])
        return np.mean(o, axis = 1)



class Beginning(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0]
        assert(len(o.shape)==2)
        return o[0, :]
    
    
class End(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0]
        assert(len(o.shape)==2)
        return o[-1, :]
    
class Mid(BaseStateConstructor):
    def process_observation(self, o_parents):
        assert(len(o_parents)) == 1
        o = o_parents[0]
        assert(len(o.shape)==2)
        i = math.ceil(o.shape[0]/2)
        return o[i, :]


class State_Wrapper:
    def __init__(self, state_constructor):
        self.state_constructor = state_constructor
        
    def __call__(self, o):
        self.state_constructor(o)
        self.state_constructor.reset_called()
        
        
def init_state_constructor(name, cfg):
    if name == "Reseau":
        # set up graph
        
        # TODO: how to normalize?
        s1 = Normalize(1, 0)
        s2 = WindowAverage(2)
        s2.set_parents([s1])

        s3 = Beginning()
        s3.set_parents([s2])
        s4 = Mid()
        s4.set_parents([s2])
        s5 = End()
        s5.set_parents([s2])

        s6 = MemoryTrace(0.5)
        s6.set_parents([s5])
        s7 = MemoryTrace(0.9)
        s7.set_parents([s5])

        s8 = Concatenate()
        s8.set_parents([s3, s4, s5, s6, s7])

        s = State_Wrapper(s8)
        
        return s
    
    else:
        s1 = Normalize(1, 0)