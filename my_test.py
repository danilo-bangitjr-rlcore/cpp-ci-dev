
import pytest
import json
import time
from fastapi.testclient import TestClient
import yaml
import tempfile

from corerl.web.app import app
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config, config_to_dict, _walk_config_and_interpolate


with open("config/opc_mountain_car_continuous.yaml", "r") as file:
    item1 = yaml.safe_load(file)


with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
    yaml.safe_dump(item1, fp, sort_keys=False, default_flow_style=False)
    fp.flush()
    config1 = direct_load_config(MainConfig, "", fp.name)

item2 = config_to_dict(MainConfig, config1)
print(item2)
item3 = _walk_config_and_interpolate(item2)

# with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
#     yaml.safe_dump(config1, fp, sort_keys=False, default_flow_style=False)
#     fp.flush()
#     config2 = direct_load_config(MainConfig, "", fp.name)
#
# item3 = config_to_dict(MainConfig, config2)

with open("req.yaml", "w") as a:
    a.write(yaml.safe_dump(item1, default_flow_style=False, sort_keys=False))

with open("res.yaml", "w") as b:
    b.write(yaml.safe_dump(item2, default_flow_style=False, sort_keys=False))
