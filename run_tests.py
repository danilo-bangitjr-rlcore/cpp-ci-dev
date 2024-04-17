import argparse
from pathlib import Path
import os
import stat
import root.utils.sweep as sweep
import logging
import subprocess
logger = logging.getLogger('test')


def main():
    parser = argparse.ArgumentParser(description="sweep maker")
    parser.add_argument('--path', default='test/configs', type=str)
    parser.add_argument('--run_file', default='main.py', type=str)
    parser.add_argument('--cmd', default='python3', type=str)
    parser.add_argument('--save_file', default='runs.sh', type=str)
    parser.add_argument('--save_path', default='test/outputs', type=str)

    cfg = parser.parse_args()

    runs = []
    for env in Path(cfg.path).iterdir():
        print(env.name)
        if '__' not in env.name:
            env_cfg_name = env.name.strip('.py')
            sweep_params = sweep.get_sweep_params(env_cfg_name, Path(cfg.path) / '{}.py'.format(env_cfg_name))
            runs += sweep_params

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    cmds = []
    with open(save_path / cfg.save_file, 'w') as f:
        for run in runs:
            write_str = '{} {} '.format(cfg.cmd, cfg.run_file)
            for k, v in run.items():
                write_str += '{}={} '.format(k, v)
            cmds.append(write_str)
            f.write(write_str + '\n')

    # make the .sh executeable
    st = os.stat(save_path / cfg.save_file)
    os.chmod(save_path / cfg.save_file, st.st_mode | stat.S_IEXEC)

    print('Starting tests...')
    for i, cmd in enumerate(cmds):
        print('Running test {}/{}'.format(i, len(cmds)))
        subprocess.call(cmd, shell=True)






if __name__ == "__main__":
    main()
