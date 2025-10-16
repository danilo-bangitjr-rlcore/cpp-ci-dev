# ruff: noqa: B008

import logging

from coregateway.proxy_utils import (
    error_responses,
    proxy_request,
)
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SuccessResponse = str

# TODO: Do we need to re define this?
class Payload(BaseModel):
    config_path: str
    coreio_id: str | None = None

coretelemetry_router = APIRouter(
    tags=["Coretelemetry Proxy"],
)

@coretelemetry_router.api_route(
    "/{path:path}",
    methods=["GET"],
    summary="Proxy requests (no body) to Coretelemetry",
    response_model=SuccessResponse,
    responses=error_responses,
)
async def proxy_no_body(path: str, request: Request):
    return await proxy_request("coretelemetry", request, path)


@coretelemetry_router.api_route(
    "/{path:path}",
    methods=["POST"],
    summary="Proxy requests (with body) to Coretelemetry",
    response_model=SuccessResponse,
    responses=error_responses,
)
async def proxy_with_body(
    path: str,
    request: Request,
    body: Payload = Body(None, description="Optional JSON body to forward"),
):
    return await proxy_request("coretelemetry", request, path, body.model_dump() if body else None)

