import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

import corerl.utils.dict as dict_u
import corerl.utils.pickle as pkl_u
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


def update_pbar(pbar: tqdm, stats: dict, keys: list) -> None:
    """
    Updates the pbar with the entries in stats for keys in keys
    """

    # keys = ['last_bellman_error', 'avg_reward']  # which information to display
    pbar_str = ''
    for k in keys:
        v = stats.get(k)
        if isinstance(v, float):
            pbar_str += '{key} : {val:.1f}, '.format(key=k, val=v)
        else:
            pbar_str += '{key} : {val} '.format(key=k, val=v)

    pbar.set_description(pbar_str)


def load_or_create[T](
    root: Path,
    cfgs: Sequence[Any],
    prefix: str,
    create_func: Callable[[], T],
) -> T:
    """
    Will either load an object or create a new one using create func. Objects are saved at root using a hash determined
    by cfgs.
    """
    cfg_hash = dict_u.hash_many(cfgs)
    save_path = root / cfg_hash / f"{prefix}-{cfg_hash}.pkl"
    obj: Any = pkl_u.maybe_load(save_path)

    if obj is not None:
        log.info(f"Loaded {prefix} from {save_path}.")
        return obj

    log.debug(f"Generating {prefix}...")
    obj = create_func()

    save_path = root / cfg_hash
    pkl_u.dump(
        save_path / f"{prefix}-{cfg_hash}.pkl",
        obj,
    )

    log.debug(f"Saved {prefix} to {save_path}.")

    return obj
