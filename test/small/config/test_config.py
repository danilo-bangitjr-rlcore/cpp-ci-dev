import importlib
from os import path

from hydra import compose, initialize
from omegaconf import OmegaConf
from pytest import fixture


@fixture
def import_module():
    def _import(module_name):
        return importlib.import_module(module_name)
    return _import

def test_initialize_config(import_module, request):
    """
    We should be able to instantiate our configuration object
    with only one import: corerl.config.

    TBD: permute and instantiate different configurations, currenly using base
    """
    import_module("corerl.config")

    root_path = request.config.rootpath
    with initialize(
        version_base=None,
        # hydra fails if config_path is not relative
        config_path=path.relpath(
            path.join(root_path, "config"),
            path.dirname(path.abspath(__file__))
        )
    ):
        cfg = compose(config_name="config")
    assert OmegaConf.to_yaml(cfg) is not None
