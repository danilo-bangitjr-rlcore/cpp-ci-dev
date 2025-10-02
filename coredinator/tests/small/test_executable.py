from pathlib import Path

import pytest

from coredinator.utils.executable import find_service_executable


def test_find_service_executable_no_matches(tmp_path: Path):
    """
    Raise FileNotFoundError when no executables found
    """
    with pytest.raises(FileNotFoundError, match="No corerl executable found"):
        find_service_executable(tmp_path, "corerl")


def test_find_service_executable_no_versioned_matches(tmp_path: Path):
    """
    Raise FileNotFoundError when executables exist but none are properly versioned
    """
    (tmp_path / "invalid-corerl-file").touch()

    with pytest.raises(FileNotFoundError, match="No valid versioned corerl executables found"):
        find_service_executable(tmp_path, "corerl")


def test_find_service_executable_single_stable(tmp_path: Path):
    """
    Find single stable version executable
    """
    exe_path = tmp_path / "linux-corerl-v0.152.0"
    exe_path.touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == exe_path


def test_find_service_executable_multiple_stable(tmp_path: Path):
    """
    Select latest stable version from multiple options
    """
    (tmp_path / "linux-corerl-v0.150.0").touch()
    (tmp_path / "linux-corerl-v0.151.0").touch()
    latest = tmp_path / "linux-corerl-v0.152.0"
    latest.touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == latest


def test_find_service_executable_prefer_stable_over_dev(tmp_path: Path):
    """
    Prefer stable versions over dev builds
    """
    stable = tmp_path / "linux-corerl-v0.152.0"
    stable.touch()
    (tmp_path / "linux-corerl-dev200-v0.153.0").touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == stable


def test_find_service_executable_only_dev_builds(tmp_path: Path):
    """
    Select latest dev build when only dev builds available
    """
    (tmp_path / "linux-corerl-dev100-v0.152.0").touch()
    latest_dev = tmp_path / "linux-corerl-dev200-v0.152.0"
    latest_dev.touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == latest_dev


def test_find_service_executable_subdirectory(tmp_path: Path):
    """
    Find executables in subdirectories
    """
    subdir = tmp_path / "dist" / "nested"
    subdir.mkdir(parents=True)
    exe_path = subdir / "linux-corerl-v0.152.0"
    exe_path.touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == exe_path


def test_find_service_executable_major_version_boundary(tmp_path: Path):
    """
    Respect major version boundary with default allow_major_upgrade=False
    """
    (tmp_path / "linux-corerl-v0.152.0").touch()
    latest = tmp_path / "linux-corerl-v1.0.0"
    latest.touch()

    result = find_service_executable(tmp_path, "corerl", allow_major_upgrade=False)
    assert result == latest


def test_find_service_executable_allow_major_upgrade(tmp_path: Path):
    """
    Allow major version upgrade when explicitly enabled
    """
    (tmp_path / "linux-corerl-v0.152.0").touch()
    latest = tmp_path / "linux-corerl-v1.0.0"
    latest.touch()

    result = find_service_executable(tmp_path, "corerl", allow_major_upgrade=True)
    assert result == latest


def test_find_service_executable_complex_scenario(tmp_path: Path):
    """
    Handle complex scenario with multiple versions and dev builds
    """
    (tmp_path / "linux-corerl-v0.150.0").touch()
    latest_stable = tmp_path / "linux-corerl-v0.152.0"
    latest_stable.touch()
    (tmp_path / "linux-corerl-dev100-v0.152.1").touch()
    (tmp_path / "linux-corerl-dev50-v1.0.0").touch()

    result = find_service_executable(tmp_path, "corerl", allow_major_upgrade=False)
    assert result == latest_stable


def test_find_service_executable_mixed_services(tmp_path: Path):
    """
    Correctly filter by service name when multiple services present
    """
    (tmp_path / "linux-coreio-v0.11.1").touch()
    corerl_exe = tmp_path / "linux-corerl-v0.152.0"
    corerl_exe.touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == corerl_exe
    assert "coreio" not in str(result)


def test_find_service_executable_patch_version_selection(tmp_path: Path):
    """
    Select highest patch version within same minor version
    """
    (tmp_path / "linux-corerl-v0.152.0").touch()
    (tmp_path / "linux-corerl-v0.152.1").touch()
    latest_patch = tmp_path / "linux-corerl-v0.152.5"
    latest_patch.touch()

    result = find_service_executable(tmp_path, "corerl")
    assert result == latest_patch
