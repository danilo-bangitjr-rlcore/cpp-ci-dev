import pytest

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config


@pytest.mark.parametrize('base,config_name', [
    ('config', 'pendulum'),
    ('config', 'saturation'),
    ('config', 'mountain_car_continuous'),
    ('config', 'opc_mountain_car_continuous'),
    ('config', 'dep_opc_mountain_car_continuous'),
    ('projects/cenovus/configs', 'offline_pretraining'),
    ('projects/drayton_valley/configs', 'drayton_valley-pilot-backwash'),
])
def test_main_configs(base: str, config_name: str):
    config = direct_load_config(MainConfig, base, config_name)
    assert isinstance(config, MainConfig)
