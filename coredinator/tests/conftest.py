import shutil
import stat
from pathlib import Path

import pytest


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary configuration file."""
    cfg = tmp_path / "example_config.yaml"
    cfg.write_text("dummy: true\n")
    return cfg

@pytest.fixture()
def dist_with_fake_executable(tmp_path: Path) -> Path:
    """Fixture to create a temporary directory with fake executables."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    src = Path(__file__).resolve().parent / "fixtures" / "fake_agent.py"
    dst_coreio = dist_dir / "coreio-1.0.0"
    dst_corerl = dist_dir / "corerl-1.0.0"
    for dst in [dst_coreio, dst_corerl]:
        shutil.copy(src, dst)
        # Make executable
        mode = dst.stat().st_mode
        dst.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dist_dir
