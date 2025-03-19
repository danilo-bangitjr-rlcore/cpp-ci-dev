import logging
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

logger = logging.getLogger(__name__)


class ConfigValidationError(BaseModel):
    kind: str
    path: str
    given_value: Any


class ConfigValidationErrors(BaseModel):
    errors: list[ConfigValidationError]



def _error_type_remapping(kind: str) -> str:
    mapping = {
        'bool_parsing': 'type_mismatch',
        'int_parsing': 'type_mismatch',
        'float_type': 'type_mismatch',
        'string_type': 'type_mismatch',
    }

    return mapping.get(kind, kind)


def _construct_error_response(details: list[ErrorDetails]) -> ConfigValidationErrors:
    errors: list[ConfigValidationError] = []

    for err_detail in details:
        errors.append(
            ConfigValidationError(
                kind=_error_type_remapping(err_detail['type']),
                path='.'.join(map(str, err_detail['loc'])),
                given_value=err_detail['input'],
            )
        )

    return ConfigValidationErrors(errors=errors)


def validate_with_error_handling[T](ta: TypeAdapter[T], config: object, context: dict[str, Any] | None = None):
    try:
        return ta.validate_python(config, context=context)

    except ValidationError as e:
        return _construct_error_response(e.errors())
