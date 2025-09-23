from collections.abc import Callable
from functools import wraps

from fastapi import HTTPException


def convert_to_http_exception(exception_type: type[Exception], status_code: int):
    """Decorator that converts specified exceptions to HTTPException with given status code.

    Example:
        @convert_to_http_exception(FileNotFoundError, status_code=400)
        def my_function():
            # FileNotFoundError will be automatically converted to HTTP 400
            open("nonexistent.txt")
    """

    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                raise HTTPException(status_code=status_code, detail=str(e)) from e

        return wrapper

    return decorator
