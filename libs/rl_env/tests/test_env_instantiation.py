import pytest

from rl_env.factory import init_env

env_names = [
    'BSM1-v0',
    'Calibration-v0',
    'DistractionWorld-v0',
    'FourRooms-v0',
    'MultiActionSaturation-v0',
    'PVS-v0',
    'Saturation-v0',
    'SaturationGoals-v0',
    'StandStillMC-v0',
    'TMaze-v0',
    'ThreeTanks-v1',
    'WindyRoom-v0',
]

@pytest.mark.parametrize('env_name', env_names)
def test_env_instantiation(env_name: str):
    env = init_env(env_name)
    assert env is not None

def test_saturation_env_with_overrides():
    overrides = {
        'decay': 0.42,
        'effect': 0.99,
        'effect_period': 123,
        'trace_val': 0.77,
    }
    env = init_env('Saturation-v0', overrides=overrides)
    assert env is not None
