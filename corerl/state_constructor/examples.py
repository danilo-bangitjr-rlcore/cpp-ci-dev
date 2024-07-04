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
        start_sc = comp.Identity()  # first component in the graph
        trace_components = []
        for trace_value in cfg.trace_values:
            # all traces will receive the output of norm_sc as input
            trace_sc = comp.MemoryTrace(trace_value, parents=[start_sc])
            trace_components.append(trace_sc)

        # finally, we will concatenate all the traces and normalized values together
        concat_parents = [start_sc] + trace_components  # the parents are normalized values and the trace's outputs
        concat_sc = comp.Concatenate(parents=concat_parents)
        self.sc = concat_sc


class AnytimeMultiTrace(CompositeStateConstructor):
    """
    A trace constructor that is composed of multiple traces
    """

    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        # define the computation graphs
        start_sc = comp.Identity()  # first component in the graph
        trace_components = []
        for trace_value in cfg.trace_values:
            # all traces will receive the output of norm_sc as input
            trace_sc = comp.MemoryTrace(trace_value, parents=[start_sc])
            trace_components.append(trace_sc)

        if cfg.representation == 'countdown':
            anytime_sc = comp.AnytimeCountDown(cfg.steps_per_decision, parents=[start_sc])
        elif cfg.representation == 'one_hot':
            anytime_sc = comp.AnytimeCountDown(cfg.steps_per_decision, parents=[start_sc])
        else:
            raise ValueError

        # finally, we will concatenate all the traces and normalized values together
        concat_parents = [start_sc] + trace_components + [
            anytime_sc]  # the parents are normalized values and the trace's outputs
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


class SimpleReseauAnytime(CompositeStateConstructor):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        identity_sc = comp.Identity()
        anytime_sc = comp.AnytimeCountDown(cfg.steps_per_decision, parents=[identity_sc])
        concat_sc = comp.Concatenate(parents=[identity_sc, anytime_sc])
        self.sc = concat_sc


class ReseauAnytime(CompositeStateConstructor):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        identity_sc = comp.Identity()

        col_sc_1 = comp.KeepCols(cfg.orp_col, parents=[identity_sc])
        col_sc_2 = comp.KeepCols(cfg.flow_rate_col, parents=[identity_sc])
        col_sc_3 = comp.KeepCols(cfg.fpm_col, parents=[identity_sc])

        # averages of the FPM
        fpm_avgs = []
        for horizon in cfg.memory:
            fpm_avgs.append(comp.LongAverage(horizon, parents=[col_sc_3]))

        # Differences in the ORP over some timeframe
        orp_diffs = []
        for horizon in cfg.memory:
            orp_diffs.append(comp.Difference(horizon, parents=[col_sc_1]))

        # Differences in the Flow Rate over some timeframe
        flow_diffs = []
        for horizon in cfg.memory:
            flow_diffs.append(comp.Difference(horizon, parents=[col_sc_2]))

        anytime_sc = comp.AnytimeCountDown(cfg.steps_per_decision, parents=[identity_sc])
        concat_parents = [identity_sc] + fpm_avgs + orp_diffs + flow_diffs + [anytime_sc]
        concat_sc = comp.Concatenate(parents=concat_parents)
        self.sc = concat_sc
