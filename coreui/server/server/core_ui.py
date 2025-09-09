# ruff: noqa: B008

import os
import sys
from typing import Any

from fastapi import Body, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse

from server.config_api.config import (
    ConfigSubfolder,
    add_tag,
    delete_tag,
    get_all_configs,
    get_config,
    get_tag,
    get_tags,
    update_tag,
)
from server.opc_api.opc_routes import opc_router


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

class ConfigsResponse(BaseModel):
    configs: list[str]

class ErrorResponse(BaseModel):
    error: str

class CoreUI:
    def __init__(self, dist_path: str | None = None):
        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:4173",
                "http://localhost:8000",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.include_router(opc_router, prefix="/api/v1/opc")

        if dist_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dist_path = os.path.join(base_dir, "client", "dist")

        meipass_path = getattr(sys, '_MEIPASS', None)
        if meipass_path:
            dist_path = os.path.join(meipass_path, "dist")

        dist_path = os.path.abspath(dist_path)
        assets_path = os.path.join(dist_path, "assets")
        if os.path.exists(assets_path):
            self.app.mount("/app/assets", StaticFiles(directory=assets_path), name="assets")

        @self.app.get("/api/health")
        async def health_check():
            return {"status": "ok"}

        @self.app.get(
            "/api/configs",
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

        @self.app.get(
            "/api/configs/{config_name}",
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

        @self.app.get(
            "/api/configs/{config_name}/tags",
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

        @self.app.get(
            "/api/configs/{config_name}/tags/{tag_name}",
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

        @self.app.post(
            "/api/configs/{config_name}/tags",
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

        @self.app.put(
            "/api/configs/{config_name}/tags/{index}",
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

        @self.app.delete(
            "/api/configs/{config_name}/tags/{index}",
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

        @self.app.get(
            "/api/raw-configs",
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

        @self.app.get(
            "/api/raw-configs/{config_name}",
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

        @self.app.get(
            "/api/raw-configs/{config_name}/tags",
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

        @self.app.get(
            "/api/raw-configs/{config_name}/tags/{tag_name}",
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

        @self.app.post(
            "/api/raw-configs/{config_name}/tags",
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

        @self.app.put(
            "/api/raw-configs/{config_name}/tags/{index}",
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

        @self.app.delete(
            "/api/raw-configs/{config_name}/tags/{index}",
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

        index_html = os.path.join(dist_path, "index.html")

        @self.app.get("/app")
        async def spa_root():
            return FileResponse(index_html)

        @self.app.get("/app/{full_path:path}")
        async def spa_catch_all(full_path: str):
            return FileResponse(index_html)

    def get_app(self):
        return self.app
