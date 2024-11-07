from corerl.eval.base_eval import BaseEval, EvalConfig
from corerl.utils.hydra import config, list_


@config('envfield', group='eval')
class EnvFieldConfig(EvalConfig):
    name: str = 'envfield'
    fields: list[str] = list_(['cached_flowrate', 'cached_height_T1'])

    offline_eval: bool = False
    online_eval: bool = True


class EnvFieldEval(BaseEval):
    def __init__(self, cfg: EnvFieldConfig, **kwargs):
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
