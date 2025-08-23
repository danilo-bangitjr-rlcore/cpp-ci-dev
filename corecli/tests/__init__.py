"""Test configuration and fixtures."""

import pytest
from click.testing import CliRunner

from corecli.main import cli as cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI runner for testing."""
    return CliRunner()
