import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lib_instrumentation.logging import LogLevel, LogRecord, get_structured_logger


def extract_json_from_logline(line: str) -> dict:
    """
    Pytest adds its own formatting to log lines, so we need to extract the JSON part
    e.g. INFO corerl:logging.py:115 {"timestamp": ..., "event": "Test message"}

    So given a line like:
        INFO corerl:logging.py:115 {"timestamp": "...", ...}
    cut out the pytest data prefix and return the parsed JSON object.
    """
    # Find the first `{` and try to parse from there
    json_start = line.find("{")
    if json_start == -1:
        raise ValueError(f"No JSON found in line: {line}")

    json_str = line[json_start:]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in line: {line}") from e

def test_log_record_creation():
    """LogRecord can be created with all required fields."""
    log = LogRecord(
        timestamp=datetime.now(UTC),
        service_name="corerl",
        level=LogLevel.INFO,
        x_correlation_id="test-correlation-id",
    )

    assert log.service_name == "corerl"
    assert log.level == LogLevel.INFO
    assert log.x_correlation_id == "test-correlation-id"


def test_log_record_validates_level():
    """LogRecord only accepts valid LogLevel enum values."""
    with pytest.raises(ValidationError):
        LogRecord(
            timestamp=datetime.now(UTC),
            service_name="corerl",
            level="INVALID_LEVEL", # type: ignore
            x_correlation_id="test-id",
        )


def test_log_record_requires_all_fields():
    """LogRecord validates that all required fields are present."""
    with pytest.raises(ValidationError):
        LogRecord(
            service_name="corerl",
            level=LogLevel.INFO,
            # type: ignore
        )


def test_structured_logger_integration(caplog: pytest.LogCaptureFixture):
    logger = get_structured_logger("corerl")

    with caplog.at_level("INFO"):
        logger.info("Test message", x_correlation_id="test-correlation-id")

    # caplog.text will contain the JSON string output
    output = caplog.text.strip()
    assert output, "Logger produced no output"

    # If multiple lines are logged, just take the last one
    json_line = output.splitlines()[-1]
    log_data = extract_json_from_logline(json_line)

    # Assertions
    assert log_data["service_name"] == "corerl"
    assert log_data["level"] == "info"
    assert log_data["x_correlation_id"] == "test-correlation-id"
    assert "timestamp" in log_data
    assert log_data["event"] == "Test message"

    datetime.fromisoformat(log_data["timestamp"])


def test_structured_logger_all_log_levels(caplog: pytest.LogCaptureFixture):
    """StructuredLogger handles all log levels correctly."""
    logger = get_structured_logger("coreio")

    with caplog.at_level("DEBUG"):
        logger.debug("Debug message", x_correlation_id="debug-id")
        logger.info("Info message", x_correlation_id="info-id")
        logger.warning("Warning message", x_correlation_id="warning-id")
        logger.error("Error message", x_correlation_id="error-id")

    output = caplog.text.strip()
    assert output, "Logger produced no output"

    log_data = output.splitlines()

    # Should have 4 log entries (debug might be filtered out depending on level)
    assert len(log_data) >= 3  # info, warning, error should always appear

    for line in log_data:
        if line:  # Skip empty lines
            log_data = extract_json_from_logline(line)
            print(log_data)
            assert log_data["service_name"] == "coreio"
            assert log_data["level"] in ["debug", "info", "warning", "error"]
            assert "x_correlation_id" in log_data
            assert "timestamp" in log_data


def test_structured_logger_auto_generates_correlation_id(caplog: pytest.LogCaptureFixture):
    """StructuredLogger automatically generates correlation IDs when not provided."""
    logger = get_structured_logger("coredinator")

    with caplog.at_level("INFO"):
        logger.info("Message without correlation ID")
        logger.info("Another message without correlation ID")

    lines = caplog.text.strip().split('\n')

    log1 = extract_json_from_logline(lines[0])
    log2 = extract_json_from_logline(lines[1])

    # Both should have correlation IDs
    assert "x_correlation_id" in log1
    assert "x_correlation_id" in log2

    # IDs should be different (UUIDs)
    assert log1["x_correlation_id"] != log2["x_correlation_id"]

    # IDs should be valid UUID format (36 characters with hyphens)
    assert len(log1["x_correlation_id"]) == 36
    assert log1["x_correlation_id"].count('-') == 4


def test_structured_logger_preserves_provided_correlation_id(caplog: pytest.LogCaptureFixture):
    """StructuredLogger uses provided correlation ID and doesn't modify it."""
    custom_id = "custom-trace-12345"
    logger = get_structured_logger("coreui")

    with caplog.at_level("WARNING"):
        logger.warning("Warning with custom ID", x_correlation_id=custom_id)

    output = caplog.text.strip()
    log_data = extract_json_from_logline(output)

    assert log_data["x_correlation_id"] == custom_id
    assert log_data["service_name"] == "coreui"
    assert log_data["level"] == "warning"

def test_multiple_loggers_independent(caplog: pytest.LogCaptureFixture):
    """Multiple logger instances operate independently."""
    logger1 = get_structured_logger("service1")
    logger2 = get_structured_logger("service2")

    with caplog.at_level("INFO"):
        logger1.info("Message from service1")
        logger2.error("Message from service2")

    lines = caplog.text.strip().split('\n')

    log1 = extract_json_from_logline(lines[0])
    log2 = extract_json_from_logline(lines[1])

    assert log1["service_name"] == "service1"
    assert log1["level"] == "info"
    assert log1["event"] == "Message from service1"

    assert log2["service_name"] == "service2"
    assert log2["level"] == "error"
    assert log2["event"] == "Message from service2"

    # Should have different correlation IDs
    assert log1["x_correlation_id"] != log2["x_correlation_id"]
