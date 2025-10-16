# ruff: noqa: B008

import logging

from coregateway.proxy_utils import (
    error_responses,
    proxy_request,
)
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel

SuccessResponse = str

class Payload(BaseModel):
    config_path: str
    coreio_id: str | None = None

coredinator_router = APIRouter(
    tags=["Coredinator Proxy"],
)

@coredinator_router.api_route(
    "/{path:path}",
    methods=["GET"],
    summary="Proxy requests (no body) to Coredinator",
    response_model=SuccessResponse,
    responses=error_responses,
)
async def proxy_no_body(path: str, request: Request):
    return await proxy_request("coredinator", request, path)


@coredinator_router.api_route(
    "/{path:path}",
    methods=["POST"],
    summary="Proxy requests (with body) to Coredinator",
    response_model=SuccessResponse,
    responses=error_responses,
)
async def proxy_with_body(
    path: str,
    request: Request,
    logger: logging.Logger = Depends(get_logger),
    body: Payload = Body(None, description="Optional JSON body to forward"),
):
    return await proxy_request("coredinator", request, path, body.model_dump() if body else None)

