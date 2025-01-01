import logging
import shutil
from pathlib import Path

# Creating env from file
import gymnasium as gym
import pandas as pd
import yaml

from corerl.configs.config import MISSING, config
from corerl.configs.loader import config_to_dict, load_config
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.environment.async_env.opc_tsdb_sim_async_env import OPCTSDBSimAsyncEnvConfig
from corerl.environment.factory import init_environment
from corerl.utils.gymnasium import gen_tag_configs_from_env


def generate_telegraf_conf(path: Path, df_ids):
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

    _logger.info(f"Generated {path}/telegraf/generated_telegraf.conf")


def generate_tag_yaml(path: Path, tags: list[TagConfig]):
    tag_path = path / "generated_tags.yaml"

    with open(tag_path, "w+") as f:
        raw_tags = config_to_dict(list[TagConfig], tags)
        yaml.safe_dump(raw_tags, f, sort_keys=False)

    _logger.info(f"Generated {tag_path}")


@config(allow_extra=True)
class Config:
    env: AsyncEnvConfig = MISSING


@load_config(Config, base="config")
def main(cfg: Config):
    assert isinstance(cfg.env, OPCTSDBSimAsyncEnvConfig), "make configs only supported for OPCTSDBSimAsyncEnvConfig"
    env: gym.Env = init_environment(cfg.env)
    _logger.info(f"Generating config with env {env}")

    tags = gen_tag_configs_from_env(env)

    string_ids = (tag.name for tag in tags)

    df_ids = pd.DataFrame(data=string_ids, columns=pd.Index(["id_name"]))
    df_ids["ns"] = cfg.env.opc_ns
    df_ids["id_type"] = "s"
    df_ids["name"] = df_ids["id_name"]
    current_path = Path(__file__).parent.absolute()

    _logger.info(f"Found {len(df_ids)} distinct nodes")
    _logger.info(df_ids)
    generate_telegraf_conf(current_path, df_ids)
    generate_tag_yaml(current_path, tags)


if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        encoding="utf-8",
        level=logging.DEBUG,
    )
    main()
