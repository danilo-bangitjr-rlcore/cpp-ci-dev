import json
from http.client import HTTPResponse
from urllib.request import urlopen

from lib_utils.maybe import Maybe
from pydantic import BaseModel, ValidationError


def maybe_validate[T: BaseModel](model: type[T], data: dict) -> Maybe[T]:
    return Maybe.from_try(lambda: model.model_validate(data), ValidationError)


def maybe_get_json[T: BaseModel](url: str, model: type[T], *, timeout: float = 5.0) -> Maybe[T]:
    return (
        maybe_open_url(url, timeout=timeout)
        .is_instance(HTTPResponse)
        .map(lambda r: r.read().decode())
        .flat_map(lambda s: Maybe.from_try(lambda: json.loads(s), json.JSONDecodeError))
        .flat_map(lambda d: maybe_validate(model, d))
    )


def maybe_open_url(url: str, timeout: float = 5.0) -> Maybe[HTTPResponse]:
    return Maybe.from_try(lambda: urlopen(url, timeout=timeout), Exception)
