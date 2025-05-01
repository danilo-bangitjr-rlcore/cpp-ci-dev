import json
import logging
from pathlib import Path

import yaml

from corerl.config import MainConfig
from corerl.configs.loader import config_to_json

log = logging.getLogger(__name__)


def prepare_save_dir(cfg: MainConfig):
    save_path = (
            Path(cfg.experiment.save_path) /
            cfg.agent_name /
            (f'seed-{cfg.experiment.seed}')
    )

    cfg_json = config_to_json(MainConfig, cfg)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yaml", "w") as f:
        yaml.safe_dump(json.loads(cfg_json), f)

    return save_path
