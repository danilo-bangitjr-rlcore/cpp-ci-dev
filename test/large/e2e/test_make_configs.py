import subprocess
import math
from os import path

import gymnasium as gym
import pytest
import yaml

from corerl.configs.loader import config_to_dict
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.gymnasium import gen_tag_configs_from_env


def assert_equal_with_tolerance(obj1, obj2, tolerance=1e-4):
    """
    Custom equality assertion with support for float comparison up to specified decimal places and list/tuple equality.
    """
    if isinstance(obj1, float) and isinstance(obj2, float):
        assert math.isclose(obj1, obj2, rel_tol=tolerance), f"Floats differ: {obj1} != {obj2}"
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        assert obj1.keys() == obj2.keys(), f"Dict keys differ: {obj1.keys()} != {obj2.keys()}"
        for key in obj1:
            assert_equal_with_tolerance(obj1[key], obj2[key], tolerance)
    elif (isinstance(obj1, list) or isinstance(obj1, tuple)) and (isinstance(obj2, list) or isinstance(obj2, tuple)):
        assert len(obj1) == len(obj2), f"List lengths differ: {len(obj1)} != {len(obj2)}"
        for item1, item2 in zip(obj1, obj2, strict=False):
            assert_equal_with_tolerance(item1, item2, tolerance)
    else:
        assert obj1 == obj2, f"Objects differ: {obj1} != {obj2}"

@pytest.mark.timeout(240)
def test_make_configs(request):
    root_path = request.config.rootpath

    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "e2e/make_configs.py",
            "--config-name",
            "opc_mountain_car_continuous",
        ],
        cwd=root_path,
    )

    proc.check_returncode()

    # assert that the generated configuration file is expected for
    # continuous mountain car

    with open(path.join(root_path, "e2e", "generated_tags.yaml")) as f:
        raw_tag_configs = yaml.safe_load(f)

    env = gym.make("MountainCarContinuous-v0")
    tag_configs = gen_tag_configs_from_env(env)
    expected_raw_tag_configs = config_to_dict(list[TagConfig], tag_configs)

    assert_equal_with_tolerance(raw_tag_configs, expected_raw_tag_configs)
