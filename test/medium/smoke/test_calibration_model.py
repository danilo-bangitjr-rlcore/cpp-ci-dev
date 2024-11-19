import sys
import pytest
import subprocess


@pytest.mark.parametrize('config', [
    'saturation',
])
@pytest.mark.timeout(120)
@pytest.mark.skipif(
    sys.platform =="win32", reason="Windows gh action runners are too flaky."
)
def test_main_configs(config: str):
    """
    Should be able to execute the main script for several configs
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """
    proc = subprocess.run([
        'uv', 'run', 'python', 'make_offline_transitions.py',
        '--config-name', f'{config}.yaml',
        'experiment.max_steps=50',
    ])
    proc.check_returncode()

    proc = subprocess.run([
        'uv', 'run', 'python', 'calibration_model.py',
        '--config-name', f'{config}.yaml',
        'data_loader=old_direct_action',
        'experiment.max_steps=25',
        'experiment.offline_steps=10',
        'experiment.cm_eval_freq=10',
        'calibration_model.train_itr=10',
        'calibration_model.num_test_rollouts=1',
        'calibration_model.max_rollout_len=30',
        'state_constructor.warmup=0',
        'offline_training=False'
    ])
    proc.check_returncode()
