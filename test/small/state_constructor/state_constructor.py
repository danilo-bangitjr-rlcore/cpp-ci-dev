from typing import Any

from corerl.state_constructor.examples import AnytimeMultiTrace, AnytimeMultiTraceConfig


def make_anytime_multi_trace(warmup: int, steps_per_decision: int):
    cfg = AnytimeMultiTraceConfig(
        trace_values=[0.9],
        warmup=warmup,
        representation='countdown',
        steps_per_decision=steps_per_decision,
        use_indicator=False,
    )
    env: Any = None
    sc = AnytimeMultiTrace(cfg, env)
    return sc
