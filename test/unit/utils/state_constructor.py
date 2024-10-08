import pytest
from omegaconf import DictConfig

from corerl.state_constructor.examples import MultiTrace, AnytimeMultiTrace


@pytest.fixture
def multi_trace():
    cfg_d = {
        'trace_values': [0.9],
        'warmup': 10,
        'name': 'multi_trace'
    }
    cfg = DictConfig(cfg_d)
    sc = MultiTrace(cfg)
    return sc


@pytest.fixture
def anytime_multi_trace():
    cfg_d = {
        'trace_values': [0.9],
        'warmup': 10,
        'name': 'anytime_multi_trace',
        'representation': 'countdown',
        'steps_per_decision': 10,  # TODO: how can I get this to be set somewhere else?
        'use_indicator': False
    }
    cfg = DictConfig(cfg_d)
    sc = AnytimeMultiTrace(cfg)
    return sc


