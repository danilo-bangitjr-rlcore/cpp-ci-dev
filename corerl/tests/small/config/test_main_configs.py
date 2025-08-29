import glob
from pathlib import Path

import pytest
from lib_config.config import MISSING
from lib_config.loader import config_to_dict, direct_load_config

from corerl.config import MainConfig


def get_bsuite_configs():
    """Get all bsuite config files from test/test/behavior directory, excluding computational_params."""
    # Get path relative to this test file
    test_file_dir = Path(__file__).parent
    behavior_dir = test_file_dir / "../../../../test/test/behavior"

    # Find all .yaml files recursively
    yaml_files = glob.glob(str(behavior_dir / "**/*.yaml"), recursive=True)

    # Convert to absolute paths and exclude computational_params
    config_paths = []
    for yaml_file in yaml_files:
        yaml_path = Path(yaml_file).resolve()
        # Remove .yaml extension for config loading
        config_name = str(yaml_path).replace(".yaml", "")
        # Exclude computational_params as it's not a behavior test
        if not config_name.endswith("computational_params"):
            config_paths.append(config_name)

    return sorted(config_paths)


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


@pytest.mark.parametrize(
    "config_name",
    [
        ("../config/bsm1"),
        ("../config/dep_mountain_car_continuous"),
        ("../projects/epcor_scrubber/configs/epcor_scrubber"),
        *get_bsuite_configs(),
    ],
)
def test_main_configs(config_name: str):
    config = direct_load_config(MainConfig, config_name=config_name)
    assert isinstance(config, MainConfig)

    # walk through config, ensure that there are no MISSING symbols or uninterpolated values
    raw_config_dict = config_to_dict(MainConfig, config)
    walk_no_missing(raw_config_dict)
