# ruff: noqa: B008

from typing import Any

from fastapi import APIRouter, Body, status
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
async def get_all_clean_configs():
    return await get_all_configs(subfolder=ConfigSubfolder.CLEAN)

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
async def get_clean_config(config_name: str):
    return await get_config(config_name, ConfigSubfolder.CLEAN)

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
async def get_clean_tags(config_name: str):
    return await get_tags(config_name, ConfigSubfolder.CLEAN)

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
async def get_clean_tag(config_name: str, tag_name: str):
    return await get_tag(config_name, tag_name, ConfigSubfolder.CLEAN)

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
async def post_clean_tag(config_name: str, tag: Tag = Body(...)):
    return await add_tag(config_name, tag.model_dump(), subfolder=ConfigSubfolder.CLEAN)

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
async def put_clean_tag(config_name: str, index: int, tag: Tag = Body(...)):
    return await update_tag(config_name, index, tag.model_dump(), ConfigSubfolder.CLEAN)

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
async def delete_clean_tag(config_name: str, index: int):
    return await delete_tag(config_name, index, ConfigSubfolder.CLEAN)

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
async def create_clean_config(config: ConfigNameRequest = Body(...)):
    return await create_config(config.config_name, subfolder=ConfigSubfolder.CLEAN)

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
async def delete_clean_config(config: ConfigNameRequest = Body(...)):
    return await delete_config(config.config_name, subfolder=ConfigSubfolder.CLEAN)


#════════════════════════════════════════════════════════════════════════════
#                          RAW CONFIG ENDPOINTS
#════════════════════════════════════════════════════════════════════════════

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
async def get_all_raw_configs():
    return await get_all_configs(subfolder=ConfigSubfolder.RAW)

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
async def get_raw_config(config_name: str):
    return await get_config(config_name, ConfigSubfolder.RAW)

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
async def get_raw_config_agent_name(config_name: str):
    return await get_config_field(config_name, "agent_name", ConfigSubfolder.RAW)

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
async def get_raw_tags(config_name: str):
    return await get_tags(config_name, ConfigSubfolder.RAW)

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
async def get_raw_tag(config_name: str, tag_name: str):
    return await get_tag(config_name, tag_name, ConfigSubfolder.RAW)

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
async def post_raw_tag(config_name: str, tag: Tag = Body(...)):
    return await add_tag(config_name, tag.model_dump(), subfolder=ConfigSubfolder.RAW)

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
async def put_raw_tag(config_name: str, index: int, tag: Tag = Body(...)):
    return await update_tag(config_name, index, tag.model_dump(), ConfigSubfolder.RAW)

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
async def delete_raw_tag(config_name: str, index: int):
    return await delete_tag(config_name, index, ConfigSubfolder.RAW)

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
async def create_raw_config(config: ConfigNameRequest = Body(...)):
    return await create_config(config.config_name, subfolder=ConfigSubfolder.RAW)

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
async def delete_raw_config(config: ConfigNameRequest = Body(...)):
    return await delete_config(config.config_name, subfolder=ConfigSubfolder.RAW)
