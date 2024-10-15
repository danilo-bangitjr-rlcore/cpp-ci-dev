import  pytest
import subprocess


@pytest.mark.parametrize('config', [
    'saturation',
])
@pytest.mark.timeout(60)
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
        'experiment.max_steps=100',
    ])
    proc.check_returncode()

    proc = subprocess.run([
        'uv', 'run', 'python', 'calibration_model.py',
        '--config-name', f'{config}.yaml',
        'experiment.max_steps=25',
        'experiment.offline_steps=25',
        'experiment.cm_eval_freq=12',
        'calibration_model.train_itr=25',
        'calibration_model.num_test_rollouts=1',
        'calibration_model.max_rollout_len=90',
        'state_constructor.warmup=0',
    ])
    proc.check_returncode()
