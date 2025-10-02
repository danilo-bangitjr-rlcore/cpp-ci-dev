"""Pytest configuration and fixtures for CoreTelemetry tests."""

import pytest


@pytest.fixture
def sample_fixture():
    """Sample fixture for future use."""
    return {"status": "ok"}
