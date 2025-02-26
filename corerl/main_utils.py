import json
import logging
from pathlib import Path

import yaml

import corerl.utils.dict as dict_u
from corerl.config import MainConfig
from corerl.configs.loader import config_to_dict, config_to_json

log = logging.getLogger(__name__)


def prepare_save_dir(cfg: MainConfig):
    dict_config = config_to_dict(MainConfig, cfg)
    if cfg.experiment.param_from_hash:
        cfg_hash = dict_u.hash(dict_config, ignore={'experiment.seed'})
        log.debug("Creating experiment param from hash:", cfg_hash)
        cfg.experiment.param = cfg_hash

    save_path = (
            Path(cfg.experiment.save_path) /
            cfg.experiment.exp_name /
            (f'param-{cfg.experiment.param}') /
            (f'seed-{cfg.experiment.seed}')
    )

    cfg_json = config_to_json(MainConfig, cfg)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yaml", "w") as f:
        yaml.safe_dump(json.loads(cfg_json), f)

    return save_path
