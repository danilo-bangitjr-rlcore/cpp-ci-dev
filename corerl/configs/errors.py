import logging
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

logger = logging.getLogger(__name__)


class ConfigValidationError(BaseModel):
    kind: str
    given_value: Any
    message: str


class ConfigValidationErrors(BaseModel):
    # schema comes from Prisma.PrismaClientKnownRequestError
    # to match client-side consumption
    name: str
    message: str
    meta: dict[str, ConfigValidationError]

def _error_type_remapping(kind: str) -> str:
    mapping = {
        'bool_parsing': 'type_mismatch',
        'int_parsing': 'type_mismatch',
        'float_parsing': 'type_mismatch',
        'string_type': 'type_mismatch',
    }

    return mapping.get(kind, kind)


def _construct_error_response(details: list[ErrorDetails]) -> ConfigValidationErrors:
    errors: dict[str, ConfigValidationError] = {}

    for err_detail in details:
        path = '.'.join(
            filter(
                lambda s: s[0].islower(),
                map(str, err_detail['loc']),
            )
        )
        errors[path] = ConfigValidationError(
            kind=_error_type_remapping(err_detail['type']),
            given_value=err_detail['input'],
            message=err_detail['msg'],
        )

    return ConfigValidationErrors(
        name='ValidationError',
        message='Failed to validate config',
        meta=errors,
    )


def validate_with_error_handling[T](ta: TypeAdapter[T], config: object, context: dict[str, Any] | None = None):
    try:
        return ta.validate_python(config, context=context)

    except ValidationError as e:
        return _construct_error_response(e.errors())

    except Exception as e:
        logger.exception(e)
        return ConfigValidationErrors(
            name='ValidationError',
            message='Failed to validate config',
            meta={},
        )
