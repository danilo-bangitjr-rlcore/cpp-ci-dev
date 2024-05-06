from corerl.state_constructor.base import CompositeStateConstructor
import corerl.state_constructor.components as comp

from omegaconf import DictConfig
import gymnasium


class MultiTrace(CompositeStateConstructor):
    """
    A trace constructor that is composed of multiple traces
    """

    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        # define the computation graphs
        nan_sc = comp.HandleNan() # first component in the graph. Handle observations with NaNs

        avg_sc = comp.Average(parents=[nan_sc]) # Average the rows

        norm_sc = comp.MaxMinNormalize(env, parents=[avg_sc])  # first component in the graph
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
        sc = comp.MaxMinNormalize(env)
        self.sc = sc


class Anytime(CompositeStateConstructor):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        identity_sc = comp.Identity()
        anytime_sc = comp.Anytime(cfg.decision_steps, parents=[identity_sc])
        concat_sc = comp.Concatenate(parents=[identity_sc, anytime_sc])
        self.sc = concat_sc

class ReseauAnytime(CompositeStateConstructor):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        identity_sc = comp.Identity()

        s0 = comp.HandleNan(parents=[identity_sc])
        s1 = comp.Average(parents=[s0])
        s2 = comp.MaxMinNormalize(env, parents=[s1])
        
        s3 = comp.KeepCols(cfg.orp_col, parents=[s2])
        s4 = comp.KeepCols(cfg.flow_rate_col, parents=[s2])
        s5 = comp.KeepCols(cfg.fpm_col, parents=[s2])
        
        # Differences in the ORP over some timeframe
        orp_diffs = []
        for horizon in cfg.memory:
            orp_diffs.append(comp.Difference(horizon, parents=[s3]))

        # Differences in the Flow Rate over some timeframe
        flow_diffs = []
        for horizon in cfg.memory:
            flow_diffs.append(comp.Difference(horizon, parents=[s4]))
        
        # averages of the FPM
        fpm_avgs = []
        for horizon in cfg.memory:
            fpm_avgs.append(comp.LongAverage(horizon, parents=[s5]))

        anytime_sc = comp.Anytime(cfg.decision_steps, parents=[identity_sc])
        concat_parents = [s2] + orp_diffs + flow_diffs + fpm_avgs + [anytime_sc]
        concat_sc = comp.Concatenate(parents=concat_parents)
        self.sc = concat_sc