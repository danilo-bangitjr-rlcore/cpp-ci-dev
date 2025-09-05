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

def _write_config_data(config_name: str, config: dict, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> None:
    config_path = _get_configs_dir(subfolder) / f"{config_name}.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    _config_cache[(config_name, subfolder)] = config

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

async def get_tags(config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    try:
        config_dict = _load_config_data(config_name, subfolder)
        tags = _get_tags(config_dict)
        return JSONResponse(content={"tags": tags}, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def get_tag(config_name: str, tag_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    try:
        config_dict = _load_config_data(config_name, subfolder)
        tags = _get_tags(config_dict)
        tag_index = _find_tag_index(tags, tag_name)
        return JSONResponse(content={"tag": tags[tag_index]}, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def get_config(config_name: str, subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    try:
        config_dict = _load_config_data(config_name, subfolder)
        return JSONResponse(content={"config": config_dict}, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def add_tag(
    config_name: str,
    tag: dict,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    try:
        config = _load_config_data(config_name, subfolder)
        tags = _get_tags(config)
        tags.append(tag["tag"])
        _write_config_data(config_name, config, subfolder)

        return JSONResponse(content={"message": "Tag created", "tag": tag["tag"], "index": len(tags) - 1},
                            status_code=status.HTTP_201_CREATED)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def update_tag(
    config_name: str,
    index: int,
    tag: dict,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    try:
        config = _load_config_data(config_name, subfolder)
        tags = _get_tags(config)
        if not (0 <= index < len(tags)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag index out of range")
        tags[index] = tag["tag"]
        _write_config_data(config_name, config, subfolder)
        return JSONResponse(content={"message": "Tag updated", "tag": tag["tag"], "index": index},
                            status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def delete_tag(
    config_name: str,
    index: int,
    subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN,
) -> JSONResponse:
    try:
        config = _load_config_data(config_name, subfolder)
        tags = _get_tags(config)
        if index < 0 or index >= len(tags):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag index out of range")
        removed = tags.pop(index)
        _write_config_data(config_name, config, subfolder)
        return JSONResponse(content={"message": "Tag deleted", "tag": removed, "index": index},
                            status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def get_all_configs(subfolder: ConfigSubfolder = ConfigSubfolder.CLEAN) -> JSONResponse:
    try:
        configs_dir = _get_configs_dir(subfolder)
        config_files = list(configs_dir.glob("*.yaml"))
        config_names = [f.stem for f in config_files]
        return JSONResponse(content={"configs": config_names},
                            status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
