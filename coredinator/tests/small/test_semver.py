from coredinator.utils.semver import (
    Version,
    find_best_version,
    parse_version_from_filename,
    sort_versions,
)


def test_parse_version_from_filename_stable():
    """
    Parse stable release versions from standard filenames
    """
    version = parse_version_from_filename("linux-corerl-v0.152.0")
    assert version == Version(major=0, minor=152, patch=0, is_dev=False, dev_build=None)


def test_parse_version_from_filename_dev_build():
    """
    Parse dev build versions with build numbers
    """
    version = parse_version_from_filename("linux-corerl-dev123-v0.152.0")
    assert version == Version(major=0, minor=152, patch=0, is_dev=True, dev_build=123)


def test_parse_version_from_filename_different_service():
    """
    Parse versions from different service names
    """
    version = parse_version_from_filename("linux-coreio-v0.11.1")
    assert version == Version(major=0, minor=11, patch=1, is_dev=False, dev_build=None)


def test_parse_version_from_filename_full_path():
    """
    Parse versions from full paths
    """
    version = parse_version_from_filename("/path/to/dist/linux-corerl-v1.2.3")
    assert version == Version(major=1, minor=2, patch=3, is_dev=False, dev_build=None)


def test_parse_version_from_filename_invalid():
    """
    Return None for invalid filenames
    """
    assert parse_version_from_filename("invalid-filename") is None
    assert parse_version_from_filename("corerl-v0.1.0") is None
    assert parse_version_from_filename("linux-corerl-0.1.0") is None
    assert parse_version_from_filename("linux-corerl-va.b.c") is None


def test_version_comparison_by_semver():
    """
    Compare versions using semantic versioning rules
    """
    v1 = Version(major=0, minor=1, patch=0, is_dev=False, dev_build=None)
    v2 = Version(major=0, minor=2, patch=0, is_dev=False, dev_build=None)
    v3 = Version(major=1, minor=0, patch=0, is_dev=False, dev_build=None)

    assert v1 < v2
    assert v2 < v3
    assert v1 < v3
    assert v3 > v2 > v1


def test_version_comparison_patch():
    """
    Compare versions with different patch levels
    """
    v1 = Version(major=0, minor=152, patch=0, is_dev=False, dev_build=None)
    v2 = Version(major=0, minor=152, patch=1, is_dev=False, dev_build=None)

    assert v1 < v2
    assert v2 > v1


def test_version_comparison_stable_vs_dev():
    """
    Stable versions should be greater than dev versions with same semver
    """
    stable = Version(major=0, minor=152, patch=0, is_dev=False, dev_build=None)
    dev = Version(major=0, minor=152, patch=0, is_dev=True, dev_build=123)

    assert dev < stable
    assert stable > dev


def test_version_comparison_dev_builds():
    """
    Compare dev builds by build number
    """
    dev1 = Version(major=0, minor=152, patch=0, is_dev=True, dev_build=100)
    dev2 = Version(major=0, minor=152, patch=0, is_dev=True, dev_build=200)

    assert dev1 < dev2
    assert dev2 > dev1


def test_version_equality():
    """
    Test version equality
    """
    v1 = Version(major=0, minor=152, patch=0, is_dev=False, dev_build=None)
    v2 = Version(major=0, minor=152, patch=0, is_dev=False, dev_build=None)

    assert v1 == v2
    assert v1 <= v2
    assert v1 >= v2


def test_sort_versions():
    """
    Sort versions from oldest to newest
    """
    versions = [
        Version(major=1, minor=0, patch=0, is_dev=False, dev_build=None),
        Version(major=0, minor=1, patch=0, is_dev=False, dev_build=None),
        Version(major=0, minor=2, patch=0, is_dev=True, dev_build=1),
        Version(major=0, minor=2, patch=0, is_dev=False, dev_build=None),
    ]

    sorted_versions = sort_versions(versions)

    assert sorted_versions[0].minor == 1
    assert sorted_versions[1].is_dev is True
    assert sorted_versions[2].minor == 2 and not sorted_versions[2].is_dev
    assert sorted_versions[3].major == 1


def test_find_best_version_empty():
    """
    Return None when no versions available
    """
    assert find_best_version([]) is None
