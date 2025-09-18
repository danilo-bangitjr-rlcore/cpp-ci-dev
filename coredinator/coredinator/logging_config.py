"""Structured logging configuration for coredinator service."""

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import structlog


def setup_structured_logging(
    log_file_path: Path | None = None,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """
    Configure structured logging with file rotation and optional console output.
    """
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[],
    )

    # Create handlers list
    handlers: list[logging.Handler] = []

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        handlers.append(console_handler)

    # Add file handler with rotation if log file path is provided
    if log_file_path:
        # Ensure parent directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        handlers.append(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            # Add correlation IDs and timestamps
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            # Format exceptions nicely
            structlog.dev.set_exc_info,
            # For file output, use JSON. For console, use dev format.
            _get_processor(log_file_path is not None),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper()),
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library root logger with our handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Force immediate flushing for file handlers
    for handler in handlers:
        if hasattr(handler, 'flush'):
            handler.flush()


def _get_processor(use_json: bool) -> Any:
    """Get the appropriate final processor for output format."""
    if use_json:
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=True)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    """
    return structlog.get_logger(name)


def flush_logs() -> None:
    """Force flush all log handlers to ensure logs are written to files."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
