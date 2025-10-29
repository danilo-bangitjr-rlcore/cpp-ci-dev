import sys
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse


class ConfigSubfolder(StrEnum):
    CLEAN = "clean"
    RAW = "raw"

_config_cache: dict[tuple[str, ConfigSubfolder], dict] = {}

def _write_config_data(path: str, config_name: str, config: dict, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> None:
    configs_dir = Path(path) / subfolder
    config_path = configs_dir / f"{config_name}.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    _config_cache[(config_name, subfolder)] = config

def _get_configs_dir(subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> Path:
    meipass = getattr(sys, '_MEIPASS', None)
    base = Path(meipass) if meipass is not None else Path(__file__).parent
    return base / "mock_configs" / subfolder

def _load_config_data(path: str, config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> dict:
    cache_key = (config_name, subfolder)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    configs_dir = Path(path) / subfolder
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file '{config_name}.yaml' not found in '{subfolder}/'",
        )

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    _config_cache[cache_key] = config
    return config

def _get_tags(config: dict) -> list[dict]:
    pipeline = config.setdefault("pipeline", {})
    # tags = pipeline.setdefault("tags", [])
    return pipeline.setdefault("tags", [])

def _find_tag_index(tags: list[dict], tag_name: str) -> int:
    for i, tag in enumerate(tags):
        if tag.get("name") == tag_name:
            return i
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tag '{tag_name}' not found",
    )

def _get_nested_value(config_dict: dict, key_path: str) -> Any:
    """Get a nested value from a dictionary using dot-separated keys, supporting lists."""
    keys = key_path.split('.')
    current = config_dict
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and key.isdigit():
            index = int(key)
            if 0 <= index < len(current):
                current = current[index]
            else:
                return None
        else:
            return None
        if current is None:
            return None
    return current

def _handle_exception(e: Exception) -> JSONResponse:
    if isinstance(e, HTTPException):
        raise e
    return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def handle_exceptions(func: Callable[..., Awaitable[JSONResponse]]) -> Callable[..., Awaitable[JSONResponse]]:
    async def wrapper(*args: Any, **kwargs: Any) -> JSONResponse:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return _handle_exception(e)
    return wrapper

@handle_exceptions
async def get_tags(path: str, config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    config_dict = _load_config_data(path, config_name, subfolder)
    tags = _get_tags(config_dict)
    return JSONResponse(content={"tags": tags}, status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_tag(path: str, config_name: str, tag_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    config_dict = _load_config_data(path, config_name, subfolder)
    tags = _get_tags(config_dict)
    tag_index = _find_tag_index(tags, tag_name)
    return JSONResponse(content={"tag": tags[tag_index]}, status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_config(path: str, config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    config_dict = _load_config_data(path, config_name, subfolder)
    return JSONResponse(content={"config": config_dict}, status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_config_field(
    path: str, config_name: str, field: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    config_dict = _load_config_data(path, config_name, subfolder)
    value = _get_nested_value(config_dict, field)
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field '{field}' not found in config '{config_name}'",
        )
    return JSONResponse(content={field: value}, status_code=status.HTTP_200_OK)

@handle_exceptions
async def add_tag(
    path: str,
    config_name: str,
    tag: dict,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    config = _load_config_data(path, config_name, subfolder)
    tags = _get_tags(config)
    tags.append(tag["tag"])
    _write_config_data(path, config_name, config, subfolder)

    return JSONResponse(content={"message": "Tag created", "tag": tag["tag"], "index": len(tags) - 1},
                        status_code=status.HTTP_201_CREATED)

@handle_exceptions
async def update_tag(
    path: str,
    config_name: str,
    index: int,
    tag: dict,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    config = _load_config_data(path, config_name, subfolder)
    tags = _get_tags(config)
    if not (0 <= index < len(tags)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag index out of range")
    tags[index] = tag["tag"]
    _write_config_data(path, config_name, config, subfolder)
    return JSONResponse(content={"message": "Tag updated", "tag": tag["tag"], "index": index},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def delete_tag(
    path: str,
    config_name: str,
    index: int,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    config = _load_config_data(path, config_name, subfolder)
    tags = _get_tags(config)
    if index < 0 or index >= len(tags):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag index out of range")
    removed = tags.pop(index)
    _write_config_data(path, config_name, config, subfolder)
    return JSONResponse(content={"message": "Tag deleted", "tag": removed, "index": index},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_all_configs(path: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    configs_dir = Path(path)/subfolder
    config_files = list(configs_dir.glob("*.yaml"))
    config_names = [f.stem for f in config_files]
    return JSONResponse(content={"configs": config_names},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def create_config(
    path: str,
    config_name: str,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    base_config = {"agent_name": config_name}
    configs_dir = Path(path) / subfolder
    config_path = configs_dir / f"{config_name}.yaml"
    if config_path.exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Config already exists")
    _write_config_data(path, config_name, base_config, subfolder)
    return JSONResponse(content={"message": "Config created", "config": base_config, "name": config_name},
                        status_code=status.HTTP_201_CREATED)

@handle_exceptions
async def delete_config(
    path: str,
    config_name: str,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    configs_dir = Path(path) / subfolder
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    config_path.unlink()
    _config_cache.pop((config_name, subfolder), None)
    return JSONResponse(content={"message": "Config deleted", "name": config_name},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_config_path(
    path: str,
    config_name: str,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    configs_dir = Path(path) / subfolder
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    return JSONResponse(content={"config_path": str(config_path.resolve())},
                        status_code=status.HTTP_200_OK)
