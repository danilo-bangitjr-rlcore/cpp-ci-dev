import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

from corerl.configs.config import MISSING, config
from corerl.configs.loader import config_to_dict, load_config
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.environment.async_env.opc_tsdb_sim_async_env import OPCTSDBSimAsyncEnvConfig
from corerl.environment.factory import init_environment
from corerl.utils.gymnasium import gen_tag_configs_from_env


@dataclass
class TagData:
    id_name: str
    name: str
    ns: int
    id_type: str = 's'


def generate_telegraf_conf(path: Path, tag_data: list[TagData]):
    _logger = logging.getLogger(__name__)
    shutil.copyfile(path / "telegraf/base_telegraf.conf", path / "telegraf/generated_telegraf.conf")
    block = ""
    with open(path / "telegraf/generated_telegraf.conf", "a") as f:
        for row in tag_data:
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

    class CustomTagYamlDumper(yaml.SafeDumper):
        pass

    def represent_float(dumper: Any, value: object):
        # round floating point numbers for serialization
        text = '{0:.4f}'.format(value).rstrip('0').rstrip('.')
        if '.' not in text:
            text += '.0'
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

    CustomTagYamlDumper.add_representer(float, represent_float)

    with open(tag_path, "w+") as f:
        raw_tags = config_to_dict(list[TagConfig], tags)
        yaml.dump(raw_tags, f, Dumper=CustomTagYamlDumper, sort_keys=False)

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
    tag_data = [
        TagData(
            id_name=tag.name,
            name=tag.name,
            ns=cfg.env.opc_ns,
        )
        for tag in tags
    ]

    current_path = Path(__file__).parent.absolute()

    _logger.info(f"Found {len(tag_data)} distinct nodes")
    generate_telegraf_conf(current_path, tag_data)
    generate_tag_yaml(current_path, tags)


if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        encoding="utf-8",
        level=logging.DEBUG,
    )
    main()
