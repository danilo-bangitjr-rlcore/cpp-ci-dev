import pytest
from omegaconf import DictConfig

from corerl.state_constructor.examples import MultiTrace, AnytimeMultiTrace


def make_anytime_multi_trace(warmup, steps_per_decision):
    cfg_d = {
        'trace_values': [0.9],
        'warmup': warmup,
        'name': 'anytime_multi_trace',
        'representation': 'countdown',
        'steps_per_decision': steps_per_decision,
        'use_indicator': False
    }
    cfg = DictConfig(cfg_d)
    sc = AnytimeMultiTrace(cfg)
    return sc
