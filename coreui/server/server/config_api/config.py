from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

_config_cache: dict[str, dict] = {}

def _write_config_data(path: str, config_name: str, config: dict) -> None:
    configs_dir = Path(path)
    config_path = configs_dir / f"{config_name}.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    _config_cache[config_name] = config

def _load_config_data(path: str, config_name: str) -> dict:
    if config_name in _config_cache:
        return _config_cache[config_name]

    configs_dir = Path(path)
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file '{config_name}.yaml' not found in {config_path}",
        )

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    _config_cache[config_name] = config
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

def _is_valid_config_file(config_file: Path) -> bool:
    """Check if a YAML file is a valid config (has agent_name field)."""
    try:
        with open(config_file, encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            return config_data is not None and "agent_name" in config_data
    except Exception:
        # Skip files that can't be parsed
        return False

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
async def get_tags(
    path: str,
    config_name: str,
) -> JSONResponse:
    config_dict = _load_config_data(path, config_name)
    tags = _get_tags(config_dict)
    return JSONResponse(
        content={"tags": tags},
        status_code=status.HTTP_200_OK,
    )

@handle_exceptions
async def get_tag(
    path: str,
    config_name: str,
    tag_name: str,
) -> JSONResponse:
    config_dict = _load_config_data(path, config_name)
    tags = _get_tags(config_dict)
    tag_index = _find_tag_index(tags, tag_name)
    return JSONResponse(
        content={"tag": tags[tag_index]},
        status_code=status.HTTP_200_OK,
    )

@handle_exceptions
async def get_config(path: str, config_name: str) -> JSONResponse:
    config_dict = _load_config_data(path, config_name)
    return JSONResponse(content={"config": config_dict}, status_code=status.HTTP_200_OK)

@handle_exceptions
async def add_tag(
    path: str,
    config_name: str,
    tag: dict,
) -> JSONResponse:
    config = _load_config_data(path, config_name)
    tags = _get_tags(config)
    tags.append(tag["tag"])
    _write_config_data(path, config_name, config)

    return JSONResponse(content={"message": "Tag created", "tag": tag["tag"], "index": len(tags) - 1},
                        status_code=status.HTTP_201_CREATED)

@handle_exceptions
async def update_tag(
    path: str,
    config_name: str,
    index: int,
    tag: dict,
) -> JSONResponse:
    config = _load_config_data(path, config_name)
    tags = _get_tags(config)
    if not (0 <= index < len(tags)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag index out of range")
    tags[index] = tag["tag"]
    _write_config_data(path, config_name, config)
    return JSONResponse(content={"message": "Tag updated", "tag": tag["tag"], "index": index},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def delete_tag(
    path: str,
    config_name: str,
    index: int,
) -> JSONResponse:
    config = _load_config_data(path, config_name)
    tags = _get_tags(config)
    if index < 0 or index >= len(tags):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag index out of range")
    removed = tags.pop(index)
    _write_config_data(path, config_name, config)
    return JSONResponse(content={"message": "Tag deleted", "tag": removed, "index": index},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_all_configs(path: str) -> JSONResponse:
    configs_dir = Path(path)
    config_files = list(configs_dir.glob("*.yaml"))

    config_names = [
        config_file.stem
        for config_file in sorted(config_files)
        if _is_valid_config_file(config_file)
    ]

    return JSONResponse(content={"configs": config_names},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def create_config(
    path: str,
    config_name: str,
) -> JSONResponse:
    base_config = {"agent_name": config_name}
    configs_dir = Path(path)
    config_path = configs_dir / f"{config_name}.yaml"
    if config_path.exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Config already exists")
    _write_config_data(path, config_name, base_config)
    return JSONResponse(content={"message": "Config created", "config": base_config, "name": config_name},
                        status_code=status.HTTP_201_CREATED)

@handle_exceptions
async def delete_config(
    path: str,
    config_name: str,
) -> JSONResponse:
    configs_dir = Path(path)
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    config_path.unlink()
    _config_cache.pop(config_name, None)
    return JSONResponse(content={"message": "Config deleted", "name": config_name},
                        status_code=status.HTTP_200_OK)

@handle_exceptions
async def get_config_path(
    path: str,
    config_name: str,
) -> JSONResponse:
    configs_dir = Path(path)
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    return JSONResponse(content={"config_path": str(config_path.resolve())},
                        status_code=status.HTTP_200_OK)
