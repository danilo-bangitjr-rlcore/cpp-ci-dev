# ruff: noqa: B008

import json
from typing import Any

import httpx
from fastapi import APIRouter, Body, HTTPException, Request, status
from pydantic import BaseModel
from server.config_api.config import (
    ConfigSubfolder,
    add_tag,
    create_config,
    delete_config,
    delete_tag,
    get_all_configs,
    get_config,
    get_config_field,
    get_config_path,
    get_tag,
    get_tags,
    update_tag,
)

config_router = APIRouter(
    tags=["Config"],
)

class Tag(BaseModel):
    tag: dict

class TagResponse(BaseModel):
    message: str
    tag: Tag
    index: int

class TagsResponse(BaseModel):
    tags: list[Tag]

class TagDetailResponse(BaseModel):
    tag: Tag

class ConfigResponse(BaseModel):
    config: dict[str, Any]

class AgentNameResponse(BaseModel):
    agent_name: str

class ConfigsResponse(BaseModel):
    configs: list[str]

class ErrorResponse(BaseModel):
    error: str

class ConfigNameRequest(BaseModel):
    config_name: str

class ConfigPathResponse(BaseModel):
    config_path: str

class AgentWithConfigResponse(BaseModel):
    agents: list[str]

InternalServerErrorResponse = str

COREGATEWAY_BASE = "http://localhost:8001"

#════════════════════════════════════════════════════════════════════════════
#                          CLEAN CONFIG ENDPOINTS
#════════════════════════════════════════════════════════════════════════════

@config_router.get(
    "/list",
    response_model=ConfigsResponse,
    responses={
        200: {
            "description": "List of all available type-validated (\"clean\") configuration names.",
            "content": {
                "application/json": {
                    "example": {
                        "configs": ["main_backwash", "secondary_config"],
                    },
                },
            },
        },
        500: {"model": ErrorResponse},
    },
    summary="List Clean Config Names",
    description="List all available type-validated (\"clean\") configuration names.",
)
async def get_all_clean_configs(request: Request):
    return await get_all_configs(path=request.app.state.config_path, subfolder=ConfigSubfolder.CLEAN)

@config_router.get(
    "/{config_name}",
    response_model=ConfigResponse,
    responses={
        200: {
            "description": "The type-validated (\"clean\") configuration as a JSON object.",
            "content": {
                "application/json": {
                    "example": {
                        "config": {
                            "agent_name": "main_backwash",
                            "infra": {},
                            "pipeline": {
                                "tags": [{"name": "DEP_BP_FLOW_SP_WA"}],
                            },
                        },
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get Clean Config",
    description="Retrieve the type-validated (\"clean\") configuration as a JSON object.",
)
async def get_clean_config(request: Request, config_name: str):
    return await get_config(request.app.state.config_path, config_name, ConfigSubfolder.CLEAN)

@config_router.get(
    "/{config_name}/tags",
    response_model=TagsResponse,
    responses={
        200: {
            "description": "All tags defined in the type-validated (\"clean\") configuration.",
            "content": {
                "application/json": {
                    "example": {
                        "tags": [{"name": "DEP_BP_FLOW_SP_WA"}],
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="List Clean Tags",
    description="List all tags defined in the type-validated (\"clean\") configuration.",
)
async def get_clean_tags(request: Request, config_name: str):
    return await get_tags(request.app.state.config_path, config_name, ConfigSubfolder.CLEAN)

@config_router.get(
    "/{config_name}/tags/{tag_name}",
    response_model=TagDetailResponse,
    responses={
        200: {
            "description": "A specific tag by name from the type-validated (\"clean\") configuration.",
            "content": {
                "application/json": {
                    "example": {
                        "tag": {"name": "DEP_BP_FLOW_SP_WA"},
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get Clean Tag",
    description="Retrieve a specific tag by name from the type-validated (\"clean\") configuration.",
)
async def get_clean_tag(request: Request, config_name: str, tag_name: str):
    return await get_tag(request.app.state.config_path, config_name, tag_name, ConfigSubfolder.CLEAN)

@config_router.post(
    "/{config_name}/tags",
    response_model=TagResponse,
    responses={
        201: {
            "description": "Tag successfully created",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tag created",
                        "tag": {"name": "NEW_TAG"},
                        "index": 2,
                    },
                },
            },
        },
        500: {"model": ErrorResponse},
    },
    summary="Add Clean Tag",
    description="Add a new tag to the type-validated (\"clean\") configuration.",
    status_code=status.HTTP_201_CREATED,
)
async def post_clean_tag(request: Request, config_name: str, tag: Tag = Body(...)):
    return await add_tag(request.app.state.config_path, config_name, tag.model_dump(), subfolder=ConfigSubfolder.CLEAN)

@config_router.put(
    "/{config_name}/tags/{index}",
    response_model=TagResponse,
    responses={
        200: {
            "description": "Tag successfully updated",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tag updated",
                        "tag": {"name": "UPDATED_TAG"},
                        "index": 2,
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Update Clean Tag",
    description="Update an existing tag by index in the type-validated (\"clean\") configuration.",
)
async def put_clean_tag(request: Request, config_name: str, index: int, tag: Tag = Body(...)):
    return await update_tag(request.app.state.config_path, config_name, index, tag.model_dump(), ConfigSubfolder.CLEAN)

@config_router.delete(
    "/{config_name}/tags/{index}",
    response_model=TagResponse,
    responses={
        200: {
            "description": "Tag successfully deleted",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tag deleted",
                        "tag": {"name": "REMOVED_TAG"},
                        "index": 2,
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Delete Clean Tag",
    description="Delete a tag by index from the type-validated (\"clean\") configuration.",
)
async def delete_clean_tag(request: Request, config_name: str, index: int):
    return await delete_tag(request.app.state.config_path, config_name, index, ConfigSubfolder.CLEAN)

@config_router.post(
    "/configs",
    response_model=ConfigResponse,
    responses={
        201: {
            "description": "Configuration successfully created",
            "content": {
                "application/json": {
                    "example": {
                        "config_name": "new_config",
                    },
                },
            },
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Create Config",
    description="Create a new configuration with the given name.",
    status_code=status.HTTP_201_CREATED,
)
async def create_clean_config(request: Request, config: ConfigNameRequest = Body(...)):
    return await create_config(request.app.state.config_path, config.config_name, subfolder=ConfigSubfolder.CLEAN)

@config_router.delete(
    "/configs",
    response_model=ConfigResponse,
    responses={
        200: {
            "description": "Configuration successfully deleted",
            "content": {
                "application/json": {
                    "example": {
                        "config_name": "new_config",
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Delete Config",
    description="Delete an existing configuration by name.",
)
async def delete_clean_config(request: Request, config: ConfigNameRequest = Body(...)):
    return await delete_config(request.app.state.config_path, config.config_name, subfolder=ConfigSubfolder.CLEAN)

@config_router.get(
    "/{config_name}/config_path",
    response_model=ConfigPathResponse,
    responses={
        200: {
            "description": "The file path of the specified configuration.",
            "content": {
                "application/json": {
                    "example": {
                        "config_path": "/path/to/configs/clean/main_backwash.yaml",
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get Config File Path",
    description="Retrieve the file path of the specified configuration.",
)
async def get_clean_config_path(request: Request, config_name: str):
    return await get_config_path(request.app.state.config_path, config_name, subfolder=ConfigSubfolder.CLEAN)

@config_router.get(
    "/agents/missing-config",
    response_model=AgentWithConfigResponse,
    responses={
        200: {
            "description": "List of agents active in coredinator but missing a clean config.",
            "content": {
                "application/json": {
                    "example": {
                        "agents": ["orphan_agent_1", "orphan_agent_2"],
                    },
                },
            },
        },
        500: {"model": InternalServerErrorResponse},
    },
    summary="Get Agents Missing Config",
    description="Retrieve agents that are active in coredinator but do not have a clean configuration available.",
)
async def get_agents_missing_config(request: Request):
    """Get agents active in coredinator but missing a clean config."""
    try:
        # Get clean configs
        clean_configs_response = await get_all_clean_configs(request)

        # Parse response body properly
        if hasattr(clean_configs_response, "body"):
            body = clean_configs_response.body
            # Convert memoryview to bytes if needed
            # Solves pyright error: Expression of type 'bytes | memoryview' cannot be assigned to declared type 'bytes'
            if isinstance(body, memoryview):
                body = bytes(body)
            clean_configs_data = json.loads(body)
        else:
            clean_configs_data = clean_configs_response

        configs = clean_configs_data.get("configs", []) if isinstance(clean_configs_data, dict) else []
        clean_configs = set(configs)

        # Get active agents from coredinator
        client: httpx.AsyncClient = request.app.state.httpx_client
        resp = await client.get(f"{COREGATEWAY_BASE}/api/agents/")
        content_type = resp.headers.get("content-type", "")
        coredinator_data = resp.json() if content_type.startswith("application/json") else resp.text

        # Extract agent names
        if isinstance(coredinator_data, dict) and "agents" in coredinator_data:
            active_agents = set(coredinator_data["agents"])
        elif isinstance(coredinator_data, list):
            active_agents = set(coredinator_data)
        else:
            active_agents = set()

        # Return agents missing config
        return {"agents": sorted(active_agents - clean_configs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents missing config: {e!s}") from e


#════════════════════════════════════════════════════════════════════════════
#                          RAW CONFIG ENDPOINTS
#════════════════════════════════════════════════════════════════════════════get_clean_config_path

@config_router.get(
    "/raw/list",
    response_model=ConfigsResponse,
    responses={
        200: {
            "description": "List of all available raw configuration names.",
            "content": {
                "application/json": {
                    "example": {
                        "configs": ["main_backwash", "secondary_config"],
                    },
                },
            },
        },
        500: {"model": ErrorResponse},
    },
    summary="List Raw Config Names",
    description="List all available raw configuration names.",
)
async def get_all_raw_configs(request: Request):
    return await get_all_configs(path=request.app.state.config_path, subfolder=ConfigSubfolder.RAW)

@config_router.get(
    "/raw/{config_name}",
    response_model=ConfigResponse,
    responses={
        200: {
            "description": "The raw (not type-validated) configuration as a JSON object.",
            "content": {
                "application/json": {
                    "example": {
                        "config": {
                            "agent_name": "main_backwash",
                            "infra": {},
                        },
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get Raw Config",
    description="Retrieve the raw (not type-validated) configuration as a JSON object.",
)
async def get_raw_config(request: Request, config_name: str):
    return await get_config(request.app.state.config_path, config_name, ConfigSubfolder.RAW)

@config_router.get(
    "/raw/{config_name}/agent_name",
    response_model=AgentNameResponse,
    responses={
        200: {
            "description": "The agent_name field from the raw configuration.",
            "content": {
                "application/json": {
                    "example": {
                        "agent_name": "main_backwash",
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get Raw Config Agent Name",
    description="Retrieve the agent_name field from the raw configuration.",
)
async def get_raw_config_agent_name(request: Request, config_name: str):
    return await get_config_field(request.app.state.config_path, config_name, "agent_name", ConfigSubfolder.RAW)

@config_router.get(
    "/raw/{config_name}/tags",
    response_model=TagsResponse,
    responses={
        200: {
            "description": "All tags defined in the raw configuration.",
            "content": {
                "application/json": {
                    "example": {
                        "tags": [{"name": "DEP_BP_FLOW_SP_WA"}],
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="List Raw Tags",
    description="List all tags defined in the raw configuration.",
)
async def get_raw_tags(request: Request, config_name: str):
    return await get_tags(request.app.state.config_path, config_name, ConfigSubfolder.RAW)

@config_router.get(
    "/raw/{config_name}/tags/{tag_name}",
    response_model=TagDetailResponse,
    responses={
        200: {
            "description": "A specific tag by name from the raw configuration.",
            "content": {
                "application/json": {
                    "example": {
                        "tag": {"name": "DEP_BP_FLOW_SP_WA"},
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get Raw Tag",
    description="Retrieve a specific tag by name from the raw configuration.",
)
async def get_raw_tag(request: Request, config_name: str, tag_name: str):
    return await get_tag(request.app.state.config_path, config_name, tag_name, ConfigSubfolder.RAW)

@config_router.post(
    "/raw/{config_name}/tags",
    response_model=TagResponse,
    responses={
        201: {
            "description": "Tag successfully created",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tag created",
                        "tag": {"name": "NEW_TAG"},
                        "index": 2,
                    },
                },
            },
        },
        500: {"model": ErrorResponse},
    },
    summary="Add Raw Tag",
    description="Add a new tag to the raw configuration.",
    status_code=status.HTTP_201_CREATED,
)
async def post_raw_tag(request: Request, config_name: str, tag: Tag = Body(...)):
    return await add_tag(request.app.state.config_path, config_name, tag.model_dump(), subfolder=ConfigSubfolder.RAW)

@config_router.put(
    "/raw/{config_name}/tags/{index}",
    response_model=TagResponse,
    responses={
        200: {
            "description": "Tag successfully updated",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tag updated",
                        "tag": {"name": "UPDATED_TAG"},
                        "index": 2,
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Update Raw Tag",
    description="Update an existing tag by index in the raw configuration.",
)
async def put_raw_tag(request: Request, config_name: str, index: int, tag: Tag = Body(...)):
    return await update_tag(request.app.state.config_path, config_name, index, tag.model_dump(), ConfigSubfolder.RAW)

@config_router.delete(
    "/raw/{config_name}/tags/{index}",
    response_model=TagResponse,
    responses={
        200: {
            "description": "Tag successfully deleted",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tag deleted",
                        "tag": {"name": "REMOVED_TAG"},
                        "index": 2,
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Delete Raw Tag",
    description="Delete a tag by index from the raw configuration.",
)
async def delete_raw_tag(request: Request, config_name: str, index: int):
    return await delete_tag(request.app.state.config_path, config_name, index, ConfigSubfolder.RAW)

@config_router.post(
    "/raw/configs",
    response_model=ConfigResponse,
    responses={
        201: {
            "description": "Configuration successfully created",
            "content": {
                "application/json": {
                    "example": {
                        "config_name": "new_config",
                    },
                },
            },
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Create Raw Config",
    description="Create a new raw configuration with the given name.",
    status_code=status.HTTP_201_CREATED,
)
async def create_raw_config(request: Request, config: ConfigNameRequest = Body(...)):
    return await create_config(request.app.state.config_path, config.config_name, subfolder=ConfigSubfolder.RAW)

@config_router.delete(
    "/raw/configs",
    response_model=ConfigResponse,
    responses={
        200: {
            "description": "Configuration successfully deleted",
            "content": {
                "application/json": {
                    "example": {
                        "config_name": "new_config",
                    },
                },
            },
        },
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Delete Raw Config",
    description="Delete an existing raw configuration by name.",
)
async def delete_raw_config(request: Request, config: ConfigNameRequest = Body(...)):
    return await delete_config(request.app.state.config_path, config.config_name, subfolder=ConfigSubfolder.RAW)
