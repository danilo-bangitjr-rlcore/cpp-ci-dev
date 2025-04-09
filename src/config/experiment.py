import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock

from utils.dict import flatten


@dataclass
class ExperimentConfig:
    name: str
    max_steps: int

    agent: dict[str, Any]
    env: dict[str, Any]


    def flatten(self):
        out = flatten(self.agent, 'agent')
        out |= flatten(self.env, 'env')

        return out


    @staticmethod
    def load(
        path: Path | str,
        overrides: list[tuple[list[str], Any]],
    ):
        path = Path(path)
        with path.open('r') as f:
            cfg = yaml.safe_load(f)

        for ks, v in overrides:
            set_nested_value(cfg, ks, v)

        return ExperimentConfig(
            name=cfg['name'],
            max_steps=cfg['max_steps'],
            agent=cfg['agent'] or {},
            env=cfg['env'] or {},
        )


def set_nested_value(dictionary: dict, keys: list, value: Any, create_missing: bool=False):
    """
    Set a value in a nested dictionary using a list of keys as the path.
    """
    current = dictionary
    for key in keys[:-1]:  # Navigate to the second-to-last key
        if key not in current:
            if not create_missing:
                raise KeyError(f"Key '{key}' not found in the nested dictionary")
            current[key] = {}
        current = current[key]

    # For the last key, also check if it exists
    if keys[-1] not in current and not create_missing:
        raise KeyError(f"Key '{keys[-1]}' not found in the nested dictionary")

    current[keys[-1]] = value


def get_next_id(db_path: Path):
    with FileLock(f'{db_path}.lock'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        try:
            cur.execute('SELECT max(id) FROM _metadata_;')
            ids = cur.fetchone()
            con.close()
            return ids[0] + 1

        except Exception:
            return 0
