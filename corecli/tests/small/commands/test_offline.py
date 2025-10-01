from click.testing import CliRunner

from corecli.main import cli


def test_offline_group_exists():
    """Test that the offline command group is registered and accessible."""
    runner = CliRunner()
    result = runner.invoke(cli, ["offline", "--help"])

    assert result.exit_code == 0
    assert "Offline RL training and analysis commands" in result.output
