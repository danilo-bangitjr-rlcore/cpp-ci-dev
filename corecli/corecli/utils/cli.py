"""CLI utility decorators and helpers."""

import functools
import logging
from collections.abc import Callable
from typing import Any

import click
from lib_utils.maybe import Maybe
from rich.console import Console

log = logging.getLogger(__name__)
console = Console()


def handle_exceptions(
    specific_exceptions: dict[type[Exception], str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ctx = Maybe.find(lambda arg: isinstance(arg, click.Context), args)

                # Check for specific exception handling
                if specific_exceptions:
                    for exc_type in specific_exceptions:
                        if isinstance(e, exc_type):
                            console.print(f"❌ Error: {e}", style="bold red")
                            ctx.tap(lambda c: c.exit(1))
                            return None

                # Log and display unhandled errors
                log.exception(f"Unexpected error in {func.__name__}")
                console.print(f"❌ Unexpected error: {e}", style="bold red")
                ctx.tap(lambda c: c.exit(1)).or_else(lambda: click.Abort())

        return wrapper

    return decorator
