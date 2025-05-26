#!/usr/bin/env python3
"""Utility script for generating telegraf and CoreRL compatible MainConfig yaml from Farama gymnasium environment."""

import argparse
import logging
import shutil
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from pprint import pformat
from typing import Any

import gymnasium as gym
import yaml

from corerl.corerl.configs.loader import config_to_dict
from corerl.corerl.data_pipeline.tag_config import TagConfig
from corerl.corerl.utils.gymnasium import gen_tag_configs_from_env

log = logging.getLogger(__name__)

@dataclass
class TagData:
    id_name: str
    name: str
    ns: int
    id_type: str = "s"


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


def generate_tag_yaml(
        path: Path, tags: list[TagConfig],
        tag_entries: list[str] | None = None,
        action_entries: list[str]| None = None,
    ):
    tag_path = path / "generated_tags.yaml"

    class CustomTagYamlDumper(yaml.SafeDumper):
        pass

    def represent_float(dumper: Any, value: object):
        # round floating point numbers for serialization
        text = "{0:.4f}".format(value).rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)

    CustomTagYamlDumper.add_representer(float, represent_float)
    CustomTagYamlDumper.add_multi_representer(
      StrEnum,
      yaml.representer.SafeRepresenter.represent_str,
  )

    def prune_tags(tags: list[dict], entries: list[str]):
        pruned_tags = [
            {key: value for key, value in tag.items() if key in entries or key == 'name'}
            for tag in tags
        ]
        return pruned_tags

    with open(tag_path, "w+") as f:
        raw_tags = config_to_dict(list[TagConfig], tags)
        action_tags = [tag for tag in raw_tags if 'action' in tag['name']]
        other_tags = [tag for tag in raw_tags if 'action' not in tag['name'] and tag['type'] != 'meta']
        meta_tags =  [tag for tag in raw_tags if tag['type'] == 'meta']

        if action_entries is not None:
            action_tags = prune_tags(action_tags, action_entries)

        if tag_entries is not None:
            other_tags = prune_tags(other_tags, tag_entries)

        pruned_tags = action_tags + other_tags + meta_tags

        yaml.dump(pruned_tags, f, Dumper=CustomTagYamlDumper, sort_keys=False)

    log.info(f"Generated {tag_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="MountainCarContinuous-v0",
        help="Value passed to gymnasium.make()",
    )
    parser.add_argument(
        "-m",
        "--meta",
        action="store_true",
        help="If specified, emits environment meta tag configurations for reward, terminated, truncated"
    )
    parser.add_argument(
        "--namespace",
        type=int,
        default=2,
        help="OPC node namespace value",
    )
    parser.add_argument(
        "--telegraf",
        action="store_true",
        help="If specified, writes generated_telegraf.conf"
    )
    parser.add_argument(
        "--tag-config",
        action="store_true",
        help="If specified, writes generated_tags.yaml"
    )
    parser.add_argument(
        "--tag-entries",
        nargs='*',
        default=[],
        help="Which entries within the tag yaml to output for tags. Use --tag-entries all to output all entries",
    )
    parser.add_argument(
        "--action-entries",
        nargs='*',
        default=[],
        help="Which additional entries within the tag yaml to output for actions. " +
        "Use --action-entries all to output all entries for actions.",
    )
    args = parser.parse_args()

    log.info(f"gym.make({repr(args.name)})")
    env: gym.Env = gym.make(args.name)
    log.info(env)

    tag_configs = gen_tag_configs_from_env(env, args.meta)
    for tag in tag_configs:
        log.debug(f"{pformat(config_to_dict(TagConfig, tag), sort_dicts=False)}")

    tag_data = [
        TagData(
            id_name=tag.name,
            name=tag.name,
            ns=args.namespace,
        )
        for tag in tag_configs
    ]

    current_path = Path(__file__).parent.absolute()

    if args.telegraf:
        generate_telegraf_conf(current_path, tag_data)

    if args.tag_entries[0] == 'all':
        tag_entries = None # do not prune
        action_entries = None # do not prune

        if len(args.action_entries) and  args.action_entries[0] != 'all':
            warnings.warn("You are specifying additional entries for actions when already outputting all entries. "
            "Ignoring.",stacklevel=0)

    elif args.action_entries[0] == 'all':
        tag_entries = args.tag_entries
        action_entries = None # do not prune
    else:
        tag_entries = args.tag_entries
        action_entries = args.tag_entries +  args.action_entries

    if args.tag_config:
        generate_tag_yaml(current_path, tag_configs, tag_entries, action_entries)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    main()
