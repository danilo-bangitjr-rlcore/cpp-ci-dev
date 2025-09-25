import logging

from corerl.agent.greedy_ac import GreedyAC
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.tags.validate_tag_configs import validate_tag_configs
from lib_config.loader import load_config

from coreoffline.config import OfflineMainConfig
from coreoffline.core.setup import create_standard_setup
from coreoffline.data_loading import (
    load_offline_transitions,
    offline_rl_from_buffer,
)
from coreoffline.offline_training.utils import get_all_offline_recommendations

log = logging.getLogger(__name__)


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    validate_tag_configs(cfg)
    save_path = cfg.save_path

    # Use create_standard_setup for standardized setup
    app_state, pipeline = create_standard_setup(cfg)

    assert isinstance(cfg.env, AsyncEnvConfig)

    column_desc = pipeline.column_descriptions
    agent = GreedyAC(cfg.agent, app_state, column_desc)

    # Offline training
    assert cfg.offline_training.offline_steps > 0
    pipeline_out, _ = load_offline_transitions(app_state, pipeline)
    assert pipeline_out is not None
    agent.update_buffer(pipeline_out)
    offline_rl_from_buffer(agent, cfg.offline_training.offline_steps)
    get_all_offline_recommendations(app_state, agent, pipeline)

    app_state.metrics.close()
    app_state.evals.close()
    agent.close()
    agent.save(save_path / 'agent')


if __name__ == "__main__":
    main()
