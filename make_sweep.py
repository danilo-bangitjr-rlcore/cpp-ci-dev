import argparse
from pathlib import Path
import os
import stat
import corerl.utils.sweep as sweep


def main():
    parser = argparse.ArgumentParser(description="sweep maker")
    parser.add_argument('--path', default='sweep/configs', type=str)
    parser.add_argument('--cfg_name', default="sweep_example", type=str)
    parser.add_argument('--run_file', default='main.py', type=str)
    parser.add_argument('--cmd', default='python3', type=str)
    parser.add_argument('--save_file', default='runs.sh', type=str)
    parser.add_argument('--save_path', default='sweep/outputs', type=str)

    cfg = parser.parse_args()
    sweep_params = sweep.get_sweep_params(cfg.cfg_name, Path(cfg.path) / '{}.py'.format(cfg.cfg_name))
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / cfg.save_file, 'w') as f:
        for run in sweep_params:
            write_str = '{} {} '.format(cfg.cmd, cfg.run_file)

            for k, v in run.items():
                write_str += "{}={} ".format(k, v)
            f.write(write_str + '\n')

    # make the .sh executeable
    st = os.stat(save_path / cfg.save_file)
    os.chmod(save_path / cfg.save_file, st.st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    main()
