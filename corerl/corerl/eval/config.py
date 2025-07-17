from functools import partial

from lib_config.config import config
from pydantic import Field

from corerl.agent.base import BaseAgent
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.agent import (
    GreedDistConfig,
    QOnlineConfig,
    QPDFPlotsConfig,
)
from corerl.eval.hindsight_return import HindsightReturnConfig
from corerl.eval.monte_carlo import MonteCarloEvalConfig
from corerl.eval.raw_data import RawDataEvalConfig, raw_data_eval
from corerl.state import AppState


@config()
class EvalConfig:
    raw_data : RawDataEvalConfig = Field(default_factory=RawDataEvalConfig)
    q_online: QOnlineConfig = Field(default_factory=QOnlineConfig)
    greed_dist_batch: GreedDistConfig = Field(default_factory=GreedDistConfig)
    greed_dist_online: GreedDistConfig = Field(default_factory=GreedDistConfig)
    q_pdf_plots: QPDFPlotsConfig = Field(default_factory=QPDFPlotsConfig)
    avg_reward: HindsightReturnConfig = Field(default_factory=HindsightReturnConfig)
    monte_carlo: MonteCarloEvalConfig = Field(default_factory=MonteCarloEvalConfig)

def register_pipeline_evals(cfg: EvalConfig, agent: BaseAgent, pipeline: Pipeline, app_state: AppState):
    pipeline.register_hook(
        cfg.raw_data.data_modes,
        cfg.raw_data.stage_codes,
        partial(raw_data_eval, cfg.raw_data, app_state),
    )
