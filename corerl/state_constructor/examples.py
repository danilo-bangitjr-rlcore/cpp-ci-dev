from dataclasses import dataclass
from corerl.state_constructor.base import CompositeStateConstructor, sc_group
import corerl.state_constructor.components as comp

import gymnasium
from corerl.utils.hydra import interpolate, list_


# ----------------
# -- MultiTrace --
# ----------------

@dataclass
class MultiTraceConfig:
    name: str = 'multi_trace'
    trace_values: list[float] = list_()
    warmup: int = 360


class MultiTrace(CompositeStateConstructor):
    """
    A trace constructor that is composed of multiple traces
    """
    def __init__(self, cfg: MultiTraceConfig, _):
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

sc_group.dispatcher(MultiTrace)

# -----------------------
# -- AnytimeMultiTrace --
# -----------------------

@dataclass
class AnytimeMultiTraceConfig(MultiTraceConfig):
    name: str = 'anytime_multi_trace'
    representation: str = 'countdown'
    steps_per_decision: int = interpolate('${interaction.steps_per_decision}')
    use_indicator: bool = True


class AnytimeMultiTrace(CompositeStateConstructor):
    """
    A trace constructor that is composed of multiple traces
    """

    def __init__(self, cfg: AnytimeMultiTraceConfig, _):
        # define the computation graphs
        start_sc = comp.Identity()  # first component in the graph
        trace_components = []
        for trace_value in cfg.trace_values:
            # all traces will receive the output of norm_sc as input
            trace_sc = comp.MemoryTrace(trace_value, parents=[start_sc])
            trace_components.append(trace_sc)

        if cfg.representation == 'countdown':
            anytime_sc = comp.AnytimeCountDown(cfg.steps_per_decision, parents=[start_sc],
                use_indicator=cfg.use_indicator)
        elif cfg.representation == 'one_hot':
            anytime_sc = comp.AnytimeOneHot(cfg.steps_per_decision, parents=[start_sc],
                use_indicator=cfg.use_indicator)
        elif cfg.representation == 'thermometer':
            anytime_sc = comp.AnytimeThermometer(cfg.steps_per_decision, parents=[start_sc],
                use_indicator=cfg.use_indicator)
        else:
            raise ValueError

        # finally, we will concatenate all the traces and normalized values together
        concat_parents = [start_sc] + trace_components + [
            anytime_sc]  # the parents are normalized values and the trace's outputs
        concat_sc = comp.Concatenate(parents=concat_parents)
        self.sc = concat_sc

sc_group.dispatcher(AnytimeMultiTrace)

# --------------
# -- Identity --
# --------------

@dataclass
class IdentityConfig:
    name: str = 'identity'

class Identity(CompositeStateConstructor):
    def __init__(self, cfg: IdentityConfig, env: gymnasium.Env):
        sc = comp.Identity()
        self.sc = sc

sc_group.dispatcher(Identity)

# ------------------
# -- SimpleReseau --
# ------------------

@dataclass
class SimpleReseauConfig:
    name: str = 'simple_reseau'
    steps_per_decision: int = interpolate('${interaction.steps_per_decision}')
    warmup: int = 180

class SimpleReseauAnytime(CompositeStateConstructor):
    def __init__(self, cfg: SimpleReseauConfig, env: gymnasium.Env):
        identity_sc = comp.Identity()
        anytime_sc = comp.AnytimeCountDown(cfg.steps_per_decision, parents=[identity_sc])
        concat_sc = comp.Concatenate(parents=[identity_sc, anytime_sc])
        self.sc = concat_sc


sc_group.dispatcher(SimpleReseauAnytime)

# -------------------
# -- ReseauAnytime --
# -------------------

@dataclass
class ReseauAnytimeConfig(SimpleReseauConfig):
    name: str = 'reseau_anytime'
    orp_col: int = 1
    flow_rate_col: int = 2
    fpm_col: int = 0
    memory: list[int] = list_([30, 60, 180, 360])
    warmup: int = 360

class ReseauAnytime(CompositeStateConstructor):
    def __init__(self, cfg: ReseauAnytimeConfig, env: gymnasium.Env):
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

        anytime_sc = comp.AnytimeOneHot(cfg.steps_per_decision, parents=[identity_sc])
        concat_parents = [identity_sc] + fpm_avgs + orp_diffs + flow_diffs + [anytime_sc]
        concat_sc = comp.Concatenate(parents=concat_parents)
        self.sc = concat_sc

sc_group.dispatcher(ReseauAnytime)
