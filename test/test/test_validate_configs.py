from pathlib import Path

import pytest
from corerl.config import MainConfig
from lib_config.config import MISSING
from lib_config.loader import config_to_dict, direct_load_config


def walk_no_missing(part: object, key_path: str = ""):
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


def get_yaml_files():
    base = Path(__file__).parent / "behavior"
    return [
        str(p)
        for p in base.rglob("*.yaml")
        if p.name != "computational_params.yaml"
    ]


@pytest.mark.parametrize("config_path", get_yaml_files())
def test_behavior_yaml_configs(config_path: str):
    config = direct_load_config(MainConfig, config_name=config_path)
    assert isinstance(config, MainConfig)
    raw_config_dict = config_to_dict(MainConfig, config)
    walk_no_missing(raw_config_dict)
