import subprocess
from os import path

import gymnasium as gym
import pytest
import yaml

from corerl.configs.loader import config_to_dict, dict_to_config
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.gymnasium import gen_tag_configs_from_env


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

    assert dict_to_config(list[TagConfig], raw_tag_configs) == dict_to_config(list[TagConfig], expected_raw_tag_configs)
