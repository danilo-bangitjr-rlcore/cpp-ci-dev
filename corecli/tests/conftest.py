import pytest
from click.testing import CliRunner

from corecli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_app():
    return cli
