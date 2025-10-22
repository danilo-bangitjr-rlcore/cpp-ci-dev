import json
from http.client import HTTPResponse
from typing import Any
from urllib.request import Request, urlopen

import click
from lib_utils.maybe import Maybe
from pydantic import BaseModel, ValidationError


def maybe_open_url(url: str, timeout: float = 5.0) -> Maybe[HTTPResponse]:
    return Maybe.from_try(lambda: urlopen(url, timeout=timeout), Exception)


def maybe_parse_json(text: str) -> Maybe[Any]:
    if not text:
        return Maybe(None)
    return Maybe.from_try(lambda: json.loads(text), json.JSONDecodeError)


def maybe_read_response(url_or_request: str | Request, timeout: float = 5.0) -> Maybe[Any]:
    return (
        Maybe.from_try(lambda: urlopen(url_or_request, timeout=timeout), Exception)
        .is_instance(HTTPResponse)
        .map(lambda r: r.read().decode())
        .flat_map(maybe_parse_json)
    )


def maybe_validate[T: BaseModel](model: type[T], data: dict) -> Maybe[T]:
    return Maybe.from_try(lambda: model.model_validate(data), ValidationError)


def maybe_get_json[T: BaseModel](url: str, model: type[T], *, timeout: float = 5.0) -> Maybe[T]:
    return (
        maybe_read_response(url, timeout=timeout)
        .flat_map(lambda d: maybe_validate(model, d))
    )


def _build_request(url: str, method: str, payload: dict[str, Any] | None):
    if method == "GET":
        return url

    return Request(
        url,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"} if payload else {},
        method=method,
    )


def _handle_request_error(port: int):
    def handler():
        click.echo(f"❌ Could not connect to coredinator on port {port}. Is it running?", err=True)
        raise click.Abort()
    return handler


def request(
    endpoint: str,
    port: int,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Maybe[Any]:
    """
    Make a request with standardized error handling.

    Returns the parsed JSON response on success, raises click.Abort on error.
    """
    url = f"http://localhost:{port}{endpoint}"
    request = _build_request(url, method, payload)

    return (
        maybe_read_response(request, timeout=timeout)
        .otherwise(lambda: _handle_request_error(port))
    )
