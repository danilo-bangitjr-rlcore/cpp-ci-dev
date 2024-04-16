import argparse
from pathlib import Path
import root.utils.sweep as sweep


def main():
    parser = argparse.ArgumentParser(description="sweep result")
    parser.add_argument('--path', default='output/experiment/', type=str)

    cfg = parser.parse_args()
    config_keys = ['agent.name', 'agent.rho']  # which entries in the configs to access
    step_keys = ['transition']  # which entries in the step logs to access
    steps = [0, 5, 9]  # which steps from the step log to retrieve

    results_list = sweep.get_sweep_results(Path(cfg.path), config_keys, steps, step_keys)

    print(sweep.list_to_df(results_list))


if __name__ == "__main__":
    main()
