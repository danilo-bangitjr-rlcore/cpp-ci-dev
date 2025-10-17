import logging
from typing import Any

import httpx
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Headers that must not be forwarded by proxies (RFC 2616 §13.5.1)
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


class NotFoundResponse(BaseModel):
    """Response model for 404 Not Found errors."""
    detail: str


class ValidationErrorDetail(BaseModel):
    """Detail structure for validation errors."""
    loc: list[Any]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Response model for 422 Validation Error."""
    detail: list[ValidationErrorDetail]


InternalServerErrorResponse = str

error_responses: dict[int | str, dict] = {
    422: {"model": ValidationErrorResponse},
    404: {"model": NotFoundResponse},
    500: {"model": InternalServerErrorResponse},
}


def clean_headers(orig: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in orig.items() if k.lower() not in HOP_BY_HOP}


def handle_proxy_exception(exc: Exception, path: str, method: str) -> HTTPException:
    """Convert httpx exceptions to appropriate FastAPI HTTPExceptions."""
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


async def proxy_request(service: str, request: Request, path: str) -> Response:
    """Core proxy logic handling all HTTP methods, body, headers, and redirect rewriting."""
    client: httpx.AsyncClient = request.app.state.httpx_client

    service_lower = service.lower()
    host = getattr(request.app.state, f"{service_lower}_host")
    port = getattr(request.app.state, f"{service_lower}_port")
    base_url = f"http://{host}:{port}"

    target_url = f"{base_url}/{path}"

    req_headers = clean_headers(dict(request.headers))
    params = dict(request.query_params)
    raw_body = await request.body()

    try:
        resp = await client.request(
            request.method,
            target_url,
            headers=req_headers,
            params=params,
            content=raw_body,
            follow_redirects=False,
        )
    except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as e:
        raise handle_proxy_exception(e, path, request.method) from e

    return Response(
        content=resp.text,
        status_code=resp.status_code,
        headers=clean_headers(dict(resp.headers)),
        media_type=resp.headers.get("content-type"),
    )
