"""Tests for the main CLI commands."""

from click.testing import CliRunner

from corecli.main import cli


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'CoreCLI - Command line tools' in result.output


def test_hello_command_default(runner: CliRunner) -> None:
    result = runner.invoke(cli, ['hello'])
    assert result.exit_code == 0
    assert 'Hello, Developer!' in result.output
    assert 'Welcome to CoreCLI' in result.output


def test_hello_command_with_name(runner: CliRunner) -> None:
    result = runner.invoke(cli, ['hello', '--name', 'Alice'])
    assert result.exit_code == 0
    assert 'Hello, Alice!' in result.output


def test_hello_command_verbose(runner: CliRunner) -> None:
    result = runner.invoke(cli, ['--verbose', 'hello', '--name', 'Bob'])
    assert result.exit_code == 0
    assert 'Hello, Bob!' in result.output
    assert 'Debug: Greeting user' in result.output


def test_version_command(runner: CliRunner) -> None:
    result = runner.invoke(cli, ['version'])
    assert result.exit_code == 0
    assert 'CoreCLI version:' in result.output
