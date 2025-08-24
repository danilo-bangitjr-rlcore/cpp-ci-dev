import pytest
from click.testing import CliRunner

# Importing `corecli.main` at collection time pulls in optional runtime
# dependencies (like `rich`) which may not be available in lightweight test
# environments. Make this import optional so focused unit tests that don't
# exercise the CLI can run without requiring those extras.
try:
    from corecli.main import cli as cli  # type: ignore
except Exception:  # pragma: no cover - keep tests runnable when deps missing
    cli = None


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI runner for testing."""
    return CliRunner()
