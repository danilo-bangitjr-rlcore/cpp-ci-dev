import glob
from pathlib import Path

from lib_utils.list import find

from coredinator.utils.semver import find_best_version, parse_version_from_filename


def find_service_executable(
    base_path: Path,
    service_name: str,
    allow_major_upgrade: bool = False,
) -> Path:
    """Find the best matching executable for a service.

    Searches for executables matching the pattern: *{service_name}-*
    and selects the most appropriate version using semantic versioning rules.
    """
    exe_pattern = str(base_path / f"**/*{service_name}-*")
    matches = glob.glob(exe_pattern, recursive=True)

    if not matches:
        raise FileNotFoundError(
            f"No {service_name} executable found in {base_path} matching '**/*{service_name}-*'",
        )

    versioned_matches = [
        (Path(match), version)
        for match in matches
        # walrus!! :=
        # assigns `version = ...` and allows the `is not None` check in one shot.
        if (version := parse_version_from_filename(Path(match).name)) is not None
    ]

    if not versioned_matches:
        raise FileNotFoundError(
            f"No valid versioned {service_name} executables found in {base_path}. "
            f"Expected format: {{platform}}-{service_name}-v{{major}}.{{minor}}.{{patch}}",
        )

    versions = [v for _, v in versioned_matches]
    best_version = find_best_version(versions, allow_major_upgrade=allow_major_upgrade)

    if best_version is None:
        raise FileNotFoundError(
            f"No suitable {service_name} version found in {base_path} with current constraints",
        )

    result = find(lambda p: p[1] == best_version, versioned_matches)
    if result is None:
        raise FileNotFoundError(f"Could not resolve executable path for {service_name}")

    return result[0]
