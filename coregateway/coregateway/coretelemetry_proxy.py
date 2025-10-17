import logging

from coregateway.proxy_utils import (
    error_responses,
    proxy_request,
)
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

coretelemetry_router = APIRouter(
    tags=["Coretelemetry Proxy"],
)

@coretelemetry_router.api_route(
    "/{path:path}",
    methods=["GET"],
    summary="Proxy requests (no body) to Coretelemetry",
    responses=error_responses,
)
async def proxy_no_body(path: str, request: Request):
    return await proxy_request("coretelemetry", request, path)


@coretelemetry_router.api_route(
    "/{path:path}",
    methods=["POST"],
    summary="Proxy requests (with body) to Coretelemetry",
    description="Accepts any JSON body and forwards it to CoreTelemetry for validation.",
    responses=error_responses,
)
async def proxy_with_body(path: str, request: Request):
    return await proxy_request("coretelemetry", request, path)

