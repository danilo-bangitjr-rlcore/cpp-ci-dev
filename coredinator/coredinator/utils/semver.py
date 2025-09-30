import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    is_dev: bool
    dev_build: int | None

    def __lt__(self, other: "Version") -> bool:
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        if self.is_dev != other.is_dev:
            return self.is_dev

        if self.is_dev and other.is_dev:
            self_build = self.dev_build if self.dev_build is not None else 0
            other_build = other.dev_build if other.dev_build is not None else 0
            return self_build < other_build

        return False

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        return not self < other


def parse_version_from_filename(filename: str) -> Version | None:
    """Parse version from executable filename.

    Expected formats:
    - {platform}-{service}-v{major}.{minor}.{patch}
    - {platform}-{service}-dev{build_number}-v{major}.{minor}.{patch}

    Examples:
    - linux-corerl-v0.152.0
    - windows-coreio-v0.11.1
    - linux-corerl-dev123-v0.152.0
    """
    pattern = r"(?:linux|windows|darwin)-[^-]+-(?:dev(\d+)-)?v(\d+)\.(\d+)\.(\d+)"
    match = re.search(pattern, filename)

    if not match:
        return None

    dev_build_str, major_str, minor_str, patch_str = match.groups()

    is_dev = dev_build_str is not None
    dev_build = int(dev_build_str) if dev_build_str else None

    return Version(
        major=int(major_str),
        minor=int(minor_str),
        patch=int(patch_str),
        is_dev=is_dev,
        dev_build=dev_build,
    )


def sort_versions(versions: list[Version]) -> list[Version]:
    """Sort versions from oldest to newest."""
    return sorted(versions)


def find_best_version(
    versions: list[Version],
    allow_major_upgrade: bool = False,
) -> Version | None:
    """Find the best version to use from a list of versions.

    Selection strategy:
    1. Prefer stable (non-dev) versions over dev versions
    2. Choose the most recent version
    3. If allow_major_upgrade is False, only consider versions with the same major version as the latest stable
    """
    if not versions:
        return None

    stable_versions = [v for v in versions if not v.is_dev]

    if not stable_versions:
        return max(versions)

    latest_stable = max(stable_versions)

    if allow_major_upgrade:
        return latest_stable

    same_major = [v for v in stable_versions if v.major == latest_stable.major]

    if not same_major:
        return None

    return max(same_major)
