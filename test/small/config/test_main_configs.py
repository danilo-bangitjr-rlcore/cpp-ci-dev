import pytest

from corerl.config import MainConfig
from corerl.configs.config import MISSING
from corerl.configs.loader import config_to_dict, direct_load_config


@pytest.mark.parametrize('base,config_name', [
    ('config', 'pendulum'),
    ('config', 'saturation'),
    ('config', 'mountain_car_continuous'),
    ('config', 'dep_mountain_car_continuous'),
    ('config', 'web_default_config'),
    ('projects/cenovus/configs', 'offline_pretraining'),
    ('projects/drayton_valley/configs', 'drayton_valley-pilot-backwash'),
    ('projects/victoria_ww/configs', 'offline_pretraining'),
    ('projects/epcor_scrubber/configs', 'epcor_scrubber'),
])
def test_main_configs(base: str, config_name: str):
    config = direct_load_config(MainConfig, base=base, config_name=config_name)
    assert isinstance(config, MainConfig)

    # walk through config, ensure that there are no MISSING symbols or uninterpolated values
    raw_config_dict = config_to_dict(MainConfig, config)

    def walk_no_missing(part: object, key_path: str=""):
        if not isinstance(part, dict):
            return

        for k, v in part.items():
            cur_key_path = k
            if key_path:
                cur_key_path = f"{key_path}.{k}"

            assert v is not MISSING, cur_key_path
            if isinstance(v, dict):
                walk_no_missing(v, cur_key_path)

            elif isinstance(v, list):
                for idx, elem in enumerate(v):
                    walk_no_missing(elem, f"{cur_key_path}[{idx}]")

    walk_no_missing(raw_config_dict)
