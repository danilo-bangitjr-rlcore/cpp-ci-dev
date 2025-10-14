class AgentMetricsException(Exception):
    """Base exception for all telemetry business logic errors.

    Attributes:
        status_code: HTTP status code to use when converting to HTTP response
        message: Human-readable error message
        context: Additional context information (agent_id, table_name, etc.)
    """

    status_code: int = 500

    def __init__(self, message: str, **context: str | int | None):
        """Initialize exception with message and optional context.

        Args:
            message: Human-readable error message
            **context: Additional context information as keyword arguments
        """
        super().__init__(message)
        self.message = message
        self.context = context


# 404 - Not Found errors
class TableNotFoundError(AgentMetricsException):
    """Raised when database table does not exist."""

    status_code = 404


class ColumnNotFoundError(AgentMetricsException):
    """Raised when database column does not exist in table."""

    status_code = 404


class NoDataFoundError(AgentMetricsException):
    """Raised when query returns no data."""

    status_code = 404


class NoMetricsAvailableError(AgentMetricsException):
    """Raised when no metrics are available for an agent."""

    status_code = 404


# 400 - Bad Request errors
class ReservedColumnError(AgentMetricsException):
    """Raised when trying to use a reserved column name as a metric."""

    status_code = 400


# 413 - Payload Too Large errors
class ResultTooLargeError(AgentMetricsException):
    """Raised when query result exceeds maximum allowed size."""

    status_code = 413


# 503 - Service Unavailable errors
class ConfigFileNotFoundError(AgentMetricsException):
    """Raised when agent configuration file cannot be found."""

    status_code = 500


class ConfigParseError(AgentMetricsException):
    """Raised when configuration file cannot be parsed."""

    status_code = 503


class DatabaseConnectionError(AgentMetricsException):
    """Raised when database connection fails."""

    status_code = 503
