from pathlib import Path
from unittest.mock import patch

import pytest

from corecli.utils.coredinator import (
    CoredinatorNotFoundError,
    find_coredinator_executable,
)


class TestFindCoredinatorExecutable:
    def test_find_executable_local_development(self) -> None:
        """Test finding executable in local development setup."""
        fake_path = Path("/fake/monorepo/coredinator/coredinator/app.py")

        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.cwd") as mock_cwd:
                mock_cwd.return_value = Path("/fake/monorepo")
                mock_exists.side_effect = lambda: str(fake_path).endswith("app.py")

                result = find_coredinator_executable()

                assert result == fake_path

    def test_find_executable_path(self) -> None:
        """Test finding executable in PATH."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("shutil.which", return_value="/usr/local/bin/coredinator"):
                result = find_coredinator_executable()

                assert result == Path("/usr/local/bin/coredinator")

    def test_find_executable_not_found(self) -> None:
        """Test when executable cannot be found."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("shutil.which", return_value=None):
                with pytest.raises(CoredinatorNotFoundError):
                    find_coredinator_executable()
