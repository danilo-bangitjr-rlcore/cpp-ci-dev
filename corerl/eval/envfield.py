from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class EnvFieldEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        self.start = True
        self.env_fields = cfg.fields
        self.cached_values = {}
        for k in self.env_fields:
            self.cached_values[k] = []

    def do_eval(self, **kwargs) -> None:
        if 'env' not in kwargs:
            raise KeyError("Missing required argument: 'env'")
        env = kwargs["env"]
        for k in self.env_fields:
            self.cached_values[k].append(getattr(env, k))

    def get_stats(self):
        return self.cached_values

    def output(self, path: Path):
        stats = self.get_stats()

        with open(path, 'w') as f:
            json.dump(stats, f)
