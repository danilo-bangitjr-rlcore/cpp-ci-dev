import subprocess

def test_saturation():
    """
    Should be able to execute the main script with the saturation experiment
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """

    proc = subprocess.run(['uv', 'run', 'python', 'main.py', '--config-name', 'saturation.yaml', 'experiment.max_steps=25'])
    proc.check_returncode()
