from root.state_constructor.base import CompositeStateConstructor
import root.state_constructor.components as comp

from omegaconf import DictConfig
import gymnasium


class MultiTrace(CompositeStateConstructor):
    """
    A trace constructor that is composed of multiple traces
    """

    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        # define the computation graphs
        norm_sc = comp.MaxminNormalize(env)  # first component in the graph
        trace_components = []
        for trace_value in cfg.trace_values:
            # all traces will receive the output of norm_sc as input
            trace_sc = comp.MemoryTrace(trace_value, parents=[norm_sc])
            trace_components.append(trace_sc)

        # finally, we will concatenate all the traces and normalized values together
        concat_parents = [norm_sc] + trace_components  # the parents are normalized values and the trace's outputs
        concat_sc = comp.Concatenate(parents=concat_parents)
        self.sc = concat_sc


class Identity(CompositeStateConstructor):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        sc = comp.Identity()
        self.sc = sc


class Normalize(CompositeStateConstructor):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        sc = comp.MaxminNormalize(env)
        self.sc = sc
