from lib_config.config import config
from pydantic import Field

from corerl.configs.eval.agent import GreedDistConfig, QOnlineConfig, QPDFPlotsConfig
from corerl.configs.eval.hindsight_return import HindsightReturnConfig
from corerl.configs.eval.monte_carlo import MonteCarloEvalConfig
from corerl.configs.eval.raw_data import RawDataEvalConfig


@config()
class EvalConfig:
    raw_data : RawDataEvalConfig = Field(default_factory=RawDataEvalConfig)
    q_online: QOnlineConfig = Field(default_factory=QOnlineConfig)
    greed_dist_batch: GreedDistConfig = Field(default_factory=GreedDistConfig)
    greed_dist_online: GreedDistConfig = Field(default_factory=GreedDistConfig)
    q_pdf_plots: QPDFPlotsConfig = Field(default_factory=QPDFPlotsConfig)
    avg_reward: HindsightReturnConfig = Field(default_factory=HindsightReturnConfig)
    monte_carlo: MonteCarloEvalConfig = Field(default_factory=MonteCarloEvalConfig)
