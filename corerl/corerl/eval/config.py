from functools import partial

from corerl.agent.base import BaseAgent
from corerl.configs.eval.config import EvalConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.raw_data import raw_data_eval
from corerl.state import AppState


def register_pipeline_evals(cfg: EvalConfig, agent: BaseAgent, pipeline: Pipeline, app_state: AppState):
    pipeline.register_hook(
        cfg.raw_data.data_modes,
        cfg.raw_data.stage_codes,
        partial(raw_data_eval, cfg.raw_data, app_state),
    )
