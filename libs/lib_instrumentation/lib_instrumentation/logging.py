import logging
import sys
import uuid
from datetime import UTC, datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class LogRecord(BaseModel):
    """Canonical log schema for all services."""

    timestamp: datetime = Field(
        description="ISO 8601 timestamp of when the log was created",
    )
    service_name: str = Field(
        description="Name of the service that generated the log",
    )
    level: LogLevel = Field(
        description="Severity level of the log entry",
    )
    x_correlation_id: str = Field(
        description="Unique identifier for tracing requests across services",
    )

    model_config = {
        "populate_by_name": True,
    }


def _configure_structlog() -> None:
    """Configure structlog with JSON output and proper formatting."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            # If log level is too low, abort pipeline and throw away log entry.
            structlog.stdlib.filter_by_level,
            # Add log level to event dict.
            structlog.stdlib.add_log_level,
            # Perform %-style formatting.
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add a timestamp in ISO 8601 format.
            structlog.processors.TimeStamper(fmt="iso"),
            # If the "stack_info" key in the event dict is true, remove it and
            # render the current stack trace in the "stack" key.
            structlog.processors.StackInfoRenderer(),
            # If the "exc_info" key in the event dict is either true or a
            # sys.exc_info() tuple, remove "exc_info" and render the exception
            # with traceback into the "exception" key.
            structlog.processors.format_exc_info,
            # If some value is in bytes, decode it to a Unicode str.
            structlog.processors.UnicodeDecoder(),
            # Render the final event dict as JSON.
            structlog.processors.JSONRenderer(),
        ],
            # `wrapper_class` is the bound logger that you get back from
            # get_logger(). This one imitates the API of `logging.Logger`.
            wrapper_class=structlog.stdlib.BoundLogger,
            # `logger_factory` is used to create wrapped loggers that are used for
            # OUTPUT. This one returns a `logging.Logger`. The final value (a JSON
            # string) from the final processor (`JSONRenderer`) will be passed to
            # the method of the same name as that you've called on the bound logger.
            logger_factory=structlog.stdlib.LoggerFactory(),
            # Effectively freeze configuration after creating the first bound
            # logger.
            cache_logger_on_first_use=True,
    )

# Configure structlog once at module load
_configure_structlog()

class StructuredLogger:
    """Wrapper for structured logging using the canonical LogRecord schema."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = structlog.get_logger(service_name)

    def _log(self, level: LogLevel, message: str, x_correlation_id: str | None = None) -> None:
        if x_correlation_id is None:
            x_correlation_id = str(uuid.uuid4())

        log_record = LogRecord(
            timestamp=datetime.now(UTC),
            service_name=self.service_name,
            level=level,
            x_correlation_id=x_correlation_id,
        )

        log_data = log_record.model_dump()

        # Use structlog's level-specific methods
        logger_method = getattr(self._logger, level.value.lower())
        logger_method(message, **log_data)

    def debug(self, message: str, x_correlation_id: str | None = None) -> None:
        self._log(LogLevel.DEBUG, message, x_correlation_id)

    def info(self, message: str, x_correlation_id: str | None = None) -> None:
        self._log(LogLevel.INFO, message, x_correlation_id)

    def warning(self, message: str, x_correlation_id: str | None = None) -> None:
        self._log(LogLevel.WARNING, message, x_correlation_id)

    def error(self, message: str, x_correlation_id: str | None = None) -> None:
        self._log(LogLevel.ERROR, message, x_correlation_id)


def get_structured_logger(service_name: str) -> StructuredLogger:
    """Get a structured logger instance for a service.

    Args:
        service_name: Name of the service (e.g., 'corerl', 'coreio', 'coredinator')

    Returns:
        StructuredLogger instance configured for the service
    """
    return StructuredLogger(service_name)

