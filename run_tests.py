import argparse
from pathlib import Path
import os
import stat
import corerl.utils.sweep as sweep
import subprocess
from omegaconf import DictConfig, OmegaConf
from main import main as run_main

import sys, os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_cfg(params: dict) -> (str, DictConfig):
    cmd = 'python3 test/output_conf.py '
    for k, v in params.items():
        cmd += '{}={} '.format(k, v)
    subprocess.call(cmd, shell=True)
    cfg = OmegaConf.load('temp-config.yaml')
    os.remove('temp-config.yaml')
    return cmd, cfg


def main():
    parser = argparse.ArgumentParser(description="sweep maker")
    parser.add_argument('--path', default='test/configs/reseau', type=str)
    parser.add_argument('--run_file', default='output_conf.py', type=str)
    parser.add_argument('--cmd', default='python3', type=str)
    parser.add_argument('--save_file', default='runs.sh', type=str)
    parser.add_argument('--save_path', default='test/outputs', type=str)

    cfg = parser.parse_args()

    run_params = []
    run_tests = []
    for env in Path(cfg.path).iterdir():
        if '__' not in env.name:
            env_cfg_name = env.name.strip('.py')

            sweep_params, tests = sweep.get_sweep_params(env_cfg_name,
                                                         Path(cfg.path) / '{}.py'.format(env_cfg_name))
            run_params += sweep_params
            run_tests += tests

    assert len(run_params) == len(run_tests)

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print('Starting tests...')
    failed_runs = []
    num_runs = len(run_tests)
    for i, param in enumerate(run_params):
        print('Running test {}/{}'.format(i, num_runs))
        cmd, cfg = get_cfg(param)
        stats = run_main(cfg)

        tests = run_tests[i]

        failed = False
        for test in tests:
            result, message = test(run_params[i], stats)
            if result is False:
                print('FAILED (run {}/{}): '.format(i, num_runs) + message)
                failed = True
        if failed:
            failed_runs.append([cmd, cfg])

    if len(failed_runs) > 0:
        print('FAILED {}/{} runs.'.format(len(failed_runs), num_runs))

        fail_path = save_path / 'failed'
        fail_path.mkdir(parents=True, exist_ok=True)

        failed_ind = 0
        with open(save_path / 'failed_runs.sh', 'w') as f:
            for cmd, cfg in failed_runs:
                f.write(cmd + '\n')
                OmegaConf.save(cfg, fail_path / "config-{}.yaml".format(failed_ind))
                failed_ind += 1
    else:
        print('All tests passed!')


if __name__ == "__main__":
    main()
