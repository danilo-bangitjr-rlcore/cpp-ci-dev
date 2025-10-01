from click.testing import CliRunner

from corecli.main import cli


def test_offline_group_exists():
    """Test that the offline command group is registered and accessible."""
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "--help"])

    assert result.exit_code == 0
    assert "Offline RL training and analysis commands" in result.output


def test_generate_tag_config_command_exists():
    """Test that generate-tag-config command is registered.

    Note: This is a smoke test that only verifies the command exists and
    shows help. Full integration testing is handled in coreoffline tests.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "generate-tag-config", "--help"])

    assert result.exit_code == 0
    assert "Generate tag configurations from database statistics" in result.output
    assert "--config" in result.output


def test_generate_tag_config_requires_config():
    """Test that generate-tag-config requires --config option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "generate-tag-config"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_train_command_exists():
    """Test that train command is registered.

    Note: This is a smoke test that only verifies the command exists and
    shows help. Full integration testing is handled in coreoffline tests.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "train", "--help"])

    assert result.exit_code == 0
    assert "Run offline RL training from data in TimescaleDB" in result.output
    assert "--config" in result.output


def test_train_requires_config():
    """Test that train requires --config option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "train"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_bclone_command_exists():
    """Test that bclone command is registered.

    Note: This is a smoke test that only verifies the command exists and
    shows help. Full integration testing is handled in coreoffline tests.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "bclone", "--help"])

    assert result.exit_code == 0
    assert "Run behaviour cloning training on offline data" in result.output
    assert "--config" in result.output


def test_bclone_requires_config():
    """Test that bclone requires --config option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "bclone"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output
