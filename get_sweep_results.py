import argparse
from pathlib import Path
import corerl.utils.sweep as sweep


def main():
    parser = argparse.ArgumentParser(description="sweep result")
    parser.add_argument('--path', default='output/experiment/', type=str)

    cfg = parser.parse_args()
    config_keys = ['agent.name', 'agent.rho']  # which entries in the configs to access
    step_keys = ['transition']  # which entries in the step logs to access
    steps = [0, 5, 9]  # which steps from the step log to retrieve
    results_list = sweep.get_sweep_results(Path(cfg.path), config_keys, steps, step_keys)
    sweep_df = sweep.list_to_df(results_list, ignore_step_logs=True)  # example of casting the list to a df

    print(sweep_df)


if __name__ == "__main__":
    main()
