from typing import Any

import pytest
from lib_config.loader import direct_load_config

from corerl.config import MainConfig


@pytest.fixture
def basic_config_path():
    return "tests/infrastructure/configs/basic_config.yaml"



def load_config(path: str, overrides: dict[str, Any]):
    cfg = direct_load_config(MainConfig, config_name=path, overrides=overrides)
    assert isinstance(cfg, MainConfig), f"Failed to load MainConfig at path {path}"
    return cfg


@pytest.fixture
def basic_config(
    basic_config_path: str,
):
    return load_config(basic_config_path, overrides={})
