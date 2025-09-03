import sys
from enum import StrEnum
from pathlib import Path

import yaml
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse


class ConfigSubfolder(StrEnum):
    CLEAN = "clean"
    RAW = "raw"

_config_cache: dict[tuple[str, ConfigSubfolder], dict] = {}

def _get_configs_dir(subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> Path:
    meipass = getattr(sys, '_MEIPASS', None)
    base = Path(meipass) if meipass is not None else Path(__file__).parent
    return base / "mock_configs" / subfolder

def _load_config_data(config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> dict:
    cache_key = (config_name, subfolder)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    config_path = _get_configs_dir(subfolder) / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file '{config_name}.yml' not found in '{subfolder}'",
        )

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    _config_cache[cache_key] = config
    return config

def _find_tag_index(tags: list[dict], tag_name: str) -> int:
    for i, tag in enumerate(tags):
        if tag.get("name") == tag_name:
            return i
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tag '{tag_name}' not found",
    )

async def get_tags(config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    config_dict = _load_config_data(config_name, subfolder)
    tags = config_dict.get("pipeline", {}).get("tags", [])
    return JSONResponse(content={"tags": tags})

async def get_tag(config_name: str, tag_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    config_dict = _load_config_data(config_name, subfolder)
    tags = config_dict.get("pipeline", {}).get("tags", [])
    tag_index = _find_tag_index(tags, tag_name)
    return JSONResponse(content={"tag": tags[tag_index]})

async def get_config(config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    config_dict = _load_config_data(config_name, subfolder)
    return JSONResponse(content={"config": config_dict})
