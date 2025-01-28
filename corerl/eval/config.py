from functools import partial

from pydantic import Field

from corerl.agent.base import BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.actor_critic import ActorCriticEvalConfig
from corerl.eval.monte_carlo import MonteCarloEvalConfig
from corerl.eval.raw_data import RawDataEvalConfig, raw_data_eval
from corerl.state import AppState


@config()
class EvalConfig:
    actor_critic: ActorCriticEvalConfig = Field(default_factory=ActorCriticEvalConfig)
    monte_carlo: MonteCarloEvalConfig = Field(default_factory=MonteCarloEvalConfig)
    raw_data : RawDataEvalConfig = Field(default_factory=RawDataEvalConfig)


def register_evals(cfg: EvalConfig, agent: BaseAgent, pipeline: Pipeline, app_state: AppState):
    pipeline.register_hook(
        cfg.raw_data.caller_codes,
        cfg.raw_data.stage_codes,
        partial(raw_data_eval, cfg.raw_data, app_state)
    )
