from dataclasses import field
from functools import partial, wraps
from typing import Callable, Concatenate

from corerl.agent.base import BaseAC
from corerl.configs.config import config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.raw_data import RawDataEvalConfig, raw_data_eval
from corerl.state import AppState


@config()
class EvalConfig:
    raw_data : RawDataEvalConfig = field(default_factory=RawDataEvalConfig)


def register_evals(cfg: EvalConfig, agent: BaseAC, pipeline: Pipeline, app_state: AppState):
    pipeline.register_hook(cfg.raw_data.caller_codes, cfg.raw_data.stage_codes, partial(raw_data_eval, cfg, app_state))


class eval_enabled[**P, R]:
    """
    Decorator that executes the function only if the specified config attribute is True.
    Args:
        config_attr (str): The name of the config attribute to check
    Returns:
        A decorator that preserves the original function's type signature
    """
    def __init__(self, config_attr: str):
        self.config_attr = config_attr

    def __call__(self, func: Callable[Concatenate[EvalConfig, P], R]) -> Callable[Concatenate[EvalConfig, P], R | None]:
        @wraps(func)
        def wrapper(cfg: EvalConfig, *args: P.args, **kwargs: P.kwargs) -> R | None:
            eval_cfg = getattr(cfg, self.config_attr)
            if eval_cfg.enabled:
                return func(eval_cfg, *args, **kwargs)
            return None
        return wrapper
