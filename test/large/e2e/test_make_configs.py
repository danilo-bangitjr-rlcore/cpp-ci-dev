import pytest
import subprocess
from os import path

import yaml


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
            "mountain_car_continuous",
        ],
        cwd=root_path,
    )

    proc.check_returncode()

    # assert that the generated configuration file is expected for
    # continuous mountain car

    with open(path.join(root_path, "e2e", "generated_tags.yaml")) as f:
        raw_tag_configs = yaml.safe_load(f)

    assert raw_tag_configs == [
        {
            "name": "reward",
            "bounds": [None, None],
            "outlier": {"name": "exp_moving", "alpha": 0.99, "tolerance": 2.0, "warmup": 10},
            "imputer": {"name": "identity"},
            "reward_constructor": [{"name": "null"}],
            "state_constructor": None,
            "is_action": False,
            "is_meta": True,
        },
        {
            "name": "terminated",
            "bounds": [None, None],
            "outlier": {"name": "exp_moving", "alpha": 0.99, "tolerance": 2.0, "warmup": 10},
            "imputer": {"name": "identity"},
            "reward_constructor": [{"name": "null"}],
            "state_constructor": None,
            "is_action": False,
            "is_meta": True,
        },
        {
            "name": "truncated",
            "bounds": [None, None],
            "outlier": {"name": "exp_moving", "alpha": 0.99, "tolerance": 2.0, "warmup": 10},
            "imputer": {"name": "identity"},
            "reward_constructor": [{"name": "null"}],
            "state_constructor": None,
            "is_action": False,
            "is_meta": True,
        },
        {
            "name": "action_0",
            "bounds": [-1.0, 1.0],
            "outlier": {"name": "exp_moving", "alpha": 0.99, "tolerance": 2.0, "warmup": 10},
            "imputer": {"name": "identity"},
            "reward_constructor": [{"name": "null"}],
            "state_constructor": None,
            "is_action": True,
            "is_meta": False,
         },
        {
            "name": "observation_0",
            "bounds": [-1.0, 1.0],
            "outlier": {"name": "exp_moving", "alpha": 0.99, "tolerance": 2.0, "warmup": 10},
            "imputer": {"name": "identity"},
            "reward_constructor": [{"name": "null"}],
            "state_constructor": None,
            "is_action": False,
            "is_meta": False,
         },
        {
            "name": "observation_1",
            "bounds": [-1.0, 1.0],
            "outlier": {"name": "exp_moving", "alpha": 0.99, "tolerance": 2.0, "warmup": 10},
            "imputer": {"name": "identity"},
            "reward_constructor": [{"name": "null"}],
            "state_constructor": None,
            "is_action": False,
            "is_meta": False,
        },
    ]
