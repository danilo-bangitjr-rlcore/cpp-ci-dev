# ruff: noqa: B008

import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Body, HTTPException, Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SuccessResponse = str

InternalServerErrorResponse = str

class NotFoundResponse(BaseModel):
    detail: str

class ValidationErrorDetail(BaseModel):
    loc: list[Any]
    msg: str
    type: str

class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail]

class Payload(BaseModel):
    config_path: str
    coreio_id: str | None = None


coredinator_router = APIRouter(
    tags=["Coredinator Proxy"],
)

# Hop-by-hop headers are only valid for a single transport-level connection
# and must not be forwarded by proxies (RFC 2616 §13.5.1).
# Examples include "Connection", "Keep-Alive", "Transfer-Encoding", etc.
# Forwarding them can break request/response handling, so we strip them out.

# Drop Content-Length: the upstream value may be wrong after proxying.

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def clean_headers(orig: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in orig.items() if k.lower() not in HOP_BY_HOP}

def handle_proxy_exception(exc: Exception, path: str, method: str) -> HTTPException:
    match exc:
        case httpx.TimeoutException():
            logger.error(f"Request timeout - path={path}, method={method}, error={exc!s}")
            return HTTPException(
                status_code=504,
                detail="Gateway timeout: upstream service did not respond in time",
            )
        case httpx.NetworkError():
            logger.error(f"Network error - path={path}, method={method}, error={exc!s}")
            return HTTPException(
                status_code=502,
                detail="Bad gateway: cannot connect to upstream service",
            )
        case httpx.HTTPError():
            logger.error(f"Upstream HTTP error - path={path}, method={method}, error={exc!s}")
            return HTTPException(
                status_code=502,
                detail="Bad gateway: upstream service error",
            )
        case _:
            logger.error(f"Unexpected error - path={path}, method={method}, error={exc!s}")
            return HTTPException(
                status_code=500,
                detail="Internal server error",
            )

async def proxy_request(request: Request, path: str, body: dict | None = None) -> Response:
    """Core proxy logic handling all HTTP methods, body, headers, and redirect rewriting."""
    client: httpx.AsyncClient = request.app.state.httpx_client
    target_url = f"{request.app.state.coredinator_base}/{path}"

    req_headers = clean_headers(dict(request.headers))
    params = dict(request.query_params)

    json_data = body if body else None
    raw_body = await request.body() if json_data is None else None

    try:
        resp = await client.request(
            request.method,
            target_url,
            headers=req_headers,
            params=params,
            json=json_data,
            content=raw_body,
            follow_redirects=False,
        )
    except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as e:
        raise handle_proxy_exception(e, path, request.method) from e

    try:
        resp_body = resp.json()
    except Exception:
        resp_body = resp.text

    return Response(
        content=resp_body if isinstance(resp_body, (str, bytes)) else json.dumps(resp_body),
        status_code=resp.status_code,
        headers=clean_headers(dict(resp.headers)),
        media_type=resp.headers.get("content-type"),
    )

@coredinator_router.api_route(
    "/{path:path}",
    methods=["GET"],
    summary="Proxy requests (no body) to Coredinator",
    response_model=SuccessResponse,
    responses={
        422: {"model": ValidationErrorResponse},
        404: {"model": NotFoundResponse},
        500: {"model": InternalServerErrorResponse},
    },
)
async def proxy_no_body(path: str, request: Request):
    return await proxy_request(request, path)


@coredinator_router.api_route(
    "/{path:path}",
    methods=["POST"],
    summary="Proxy requests (with body) to Coredinator",
    response_model=SuccessResponse,
    responses={
        422: {"model": ValidationErrorResponse},
        404: {"model": NotFoundResponse},
        500: {"model": InternalServerErrorResponse},
    },
)
async def proxy_with_body(
    path: str,
    request: Request,
    body: Payload = Body(None, description="Optional JSON body to forward"),
):
    return await proxy_request(request, path, body.model_dump() if body else None)

