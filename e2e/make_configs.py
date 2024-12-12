import pandas as pd
from pathlib import Path
import shutil
import logging

# Creating env from file
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from itertools import chain

# Imports from main
from corerl.environment.factory import init_environment
from corerl.config import MainConfig  # noqa: F401
from corerl.interaction.anytime_interaction import AnytimeInteraction  # noqa: F401
from corerl.utils.device import device  # noqa: F401
from corerl.agent.factory import init_agent  # noqa: F401
from corerl.state_constructor.factory import init_state_constructor  # noqa: F401
from corerl.data_pipeline.factory import init_data_loader  # noqa: F401
from corerl.data.factory import init_transition_creator  # noqa: F401
from corerl.utils.plotting import make_online_plots, make_offline_plots  # noqa: F401
from corerl.eval.composite_eval import CompositeEval  # noqa: F401
from corerl.data_pipeline.base import BaseDataLoader, OldBaseDataLoader  # noqa: F401
from corerl.data_pipeline.direct_action import DirectActionDataLoader, OldDirectActionDataLoader  # noqa: F401
from corerl.environment.reward.factory import init_reward_function  # noqa: F401
from corerl.data_pipeline.datatypes import OldObsTransition, Transition, ObsTransition, Trajectory  # noqa: F401
from corerl.interaction.base import BaseInteraction  # noqa: F401
from corerl.data.obs_normalizer import ObsTransitionNormalizer  # noqa: F401
from corerl.data.transition_normalizer import TransitionNormalizer  # noqa: F401
from corerl.alerts.composite_alert import CompositeAlert  # noqa: F401
from corerl.data.transition_creator import OldAnytimeTransitionCreator, BaseTransitionCreator  # noqa: F401
from corerl.state_constructor.base import BaseStateConstructor  # noqa: F401
from corerl.agent.base import BaseAgent  # noqa: F401
from corerl.utils.plotting import make_actor_critic_plots, make_reseau_gvf_critic_plot  # noqa: F401
from corerl.data_pipeline.transition_load_funcs import make_transitions  # noqa: F401
import corerl.utils.dict as dict_u  # noqa: F401
import corerl.utils.nullable as nullable  # noqa: F401

from corerl.utils.gymnasium import gen_tag_configs_from_env
from corerl.data_pipeline.tag_config import TagConfig

def generate_telegraf_conf(path: Path, df_ids):
    _logger = logging.getLogger(__name__)
    shutil.copyfile(path / "telegraf/base_telegraf.conf", path / "telegraf/generated_telegraf.conf")
    block = ""
    with open(path / "telegraf/generated_telegraf.conf", "a") as f:
        for row in df_ids.itertuples():
            block += "[[inputs.opcua.nodes]]\n"
            block += " " * 2 + f'namespace = "{row.ns}"\n'
            block += " " * 2 + f'identifier_type = "{row.id_type}"\n'
            block += " " * 2 + f'identifier = "{row.id_name}"\n'
            block += " " * 2 + 'name = "val"\n'
            block += " " * 2 + f'default_tags = {{ name = "{row.name}" }}\n'
            block += "\n"
        f.write(block)

    _logger.info(f"Generetad {path}/telegraf/generated_telegraf.conf")


def generate_omegaconf_yaml(path: Path, tags: dict[str, list[TagConfig]]):
    omegaconf_fp = path / "generated_tags.yaml"
    with open(omegaconf_fp, "+w") as f:
        OmegaConf.save(tags, f)
    _logger.info(f"Generated {omegaconf_fp}")


@hydra.main(version_base=None, config_name='config', config_path="../config/")
def main(cfg: DictConfig):
    env: gym.Env = init_environment(cfg.env)
    _logger.info(f"Generating config with env {env}")

    tags = gen_tag_configs_from_env(env)

    ns = 2
    if "ns" in cfg:
        ns = cfg.env.ns

    string_ids = (tag.name for tag in chain.from_iterable(tags.values()))

    df_ids = pd.DataFrame(data=string_ids, columns=pd.Index(["id_name"]))
    df_ids["ns"] = ns
    df_ids["id_type"] = "s"
    df_ids["name"] = df_ids["id_name"]
    current_path = Path(__file__).parent.absolute()

    _logger.info(f"Found {len(df_ids)} distinct nodes")
    _logger.info(df_ids)
    generate_telegraf_conf(current_path, df_ids)
    generate_omegaconf_yaml(current_path, tags)


if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    main()
