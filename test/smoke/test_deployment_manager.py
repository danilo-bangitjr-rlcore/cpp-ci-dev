import pytest
import subprocess


@pytest.mark.timeout(60)
def test_deployment_manager():
    """
    Should be able to execute the deployment script with the saturation experiment
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """

    proc = subprocess.run([
        'uv', 'run',
        'python', 'deployment_manager.py',
        '--config-name', 'deployment_manager.yaml',
    ])
    proc.check_returncode()
