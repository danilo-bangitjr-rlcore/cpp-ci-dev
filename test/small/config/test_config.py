import importlib
from typing import Any
from pytest import fixture
from collections.abc import Callable


@fixture
def import_module():
    def _import(module_name: str):
        return importlib.import_module(module_name)
    return _import

def test_initialize_config(import_module: Callable[[str], Any]):
    """
    We should be able to instantiate our configuration object
    with only one import: corerl.config.

    TBD: permute and instantiate different configurations, currenly using base
    """
    config_module = import_module("corerl.config")
    cfg = config_module.MainConfig()

    # a bit of a tautology, but forces the config to be
    # fully constructed
    assert isinstance(cfg, config_module.MainConfig)
