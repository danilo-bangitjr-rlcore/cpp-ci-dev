import subprocess

import pytest


@pytest.mark.timeout(60)
def test_main_smoke():
    """
    Should be able to execute the main script without error.
    If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """
    proc = subprocess.run([
        'python', 'src/main.py',
        '-e', 'tests/assets/base_gac.yaml',
        '-s', '0',
    ], check=False)
    proc.check_returncode()
