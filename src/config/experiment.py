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
    def load(path: Path | str):
        path = Path(path)
        with path.open('r') as f:
            cfg = yaml.safe_load(f)

        return ExperimentConfig(
            name=cfg['name'],
            max_steps=cfg['max_steps'],
            agent=cfg['agent'] or {},
            env=cfg['env'] or {},
        )


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
