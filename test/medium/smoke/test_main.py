import  pytest
import subprocess


@pytest.mark.parametrize('config', [
    'pendulum',
    'saturation',
])
@pytest.mark.timeout(120)
def test_main_configs(config: str):
    """
    Should be able to execute the main script for several configs
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """

    proc = subprocess.run(['uv', 'run', 'python', 'main.py', '--config-name', f'{config}.yaml', 'experiment.max_steps=25'])
    proc.check_returncode()
