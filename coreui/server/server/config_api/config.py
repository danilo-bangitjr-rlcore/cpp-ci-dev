import json
import sys
from pathlib import Path

import yaml
from corerl.config import MainConfig
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import config_from_dict, config_to_json

_config_cache: dict[str, MainConfig] = {}

def _get_configs_dir() -> Path:
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass is not None:
        return Path(meipass) / "mock_configs"
    return Path(__file__).parent / "mock_configs"

CONFIGS_DIR = _get_configs_dir()

def _validate_config(data: dict) -> MainConfig:
    result = config_from_dict(MainConfig, data)
    if isinstance(result, ConfigValidationErrors):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=json.loads(config_to_json(ConfigValidationErrors, result)),
        )
    return result

def _load_config_data(config_name: str) -> dict | MainConfig:
    if config_name in _config_cache:
        return _config_cache[config_name]

    config_path = CONFIGS_DIR / f"{config_name}.yml"
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file '{config_name}.yml' not found",
        )

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    _config_cache[config_name] = config
    return config


def _find_tag_index(tags: list[dict], tag_name: str) -> int:
    for i, tag in enumerate(tags):
        if tag.get("name") == tag_name:
            return i
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tag '{tag_name}' not found",
    )

async def get_tags(config_name: str) -> JSONResponse:
    config_dict = _load_config_data(config_name)
    if isinstance(config_dict, MainConfig):
        config_dict = json.loads(config_to_json(MainConfig, config_dict))
    tags = config_dict.get("pipeline", {}).get("tags", [])
    return JSONResponse(content={"tags": tags})

async def get_tag(config_name: str, tag_name: str) -> JSONResponse:
    config_dict = _load_config_data(config_name)
    if isinstance(config_dict, MainConfig):
        config_dict = json.loads(config_to_json(MainConfig, config_dict))
    tags = config_dict.get("pipeline", {}).get("tags", [])

    tag_index = _find_tag_index(tags, tag_name)
    return JSONResponse(content={"tag": tags[tag_index]})

async def get_config(config_name: str) -> JSONResponse:
    config_dict = _load_config_data(config_name)
    return JSONResponse(content={"config": config_dict})
