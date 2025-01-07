import importlib.util
from pathlib import Path
from typing import Callable

def add_key_to_run(run: dict, key: str, values: list):
    run_list = []
    for value in values:
        if isinstance(value, list):
            value = '\\[' + ','.join(value) + '\\]'  # so hydra understands this value as a list

        run_ = run.copy()
        run_[key] = value
        run_list.append(run_)
    return run_list


def params_to_list(params: dict) -> list[dict] | tuple[list[dict], list[dict]]:
    runs = []
    seeds = [0]
    for new_key, new_values in params.items():
        if new_key == 'experiment.seed':
            seeds = new_values
        else:
            new_runs = []
            if len(runs) == 0:
                for v in new_values:
                    run_ = {new_key: v}
                    new_runs.append(run_)

            else:
                for run in runs:
                    if isinstance(new_values, list):  # add to all runs
                        new_runs += add_key_to_run(run, new_key, new_values)
                    elif isinstance(new_values, Callable):
                        if new_values(run) is not None:
                            new_values_ = new_values(run)
                            new_runs += add_key_to_run(run, new_key, new_values_)
                        else:
                            new_runs.append(run)

            runs = new_runs

    runs_with_seeds = []
    for i in range(len(runs)):
        for seed in seeds:
            seed_run = runs[i].copy()
            seed_run['experiment.seed'] = seed
            seed_run['experiment.param'] = i
            runs_with_seeds.append(seed_run)

    if 'tests' in params.keys():
        expected_results = [params['tests'] for i in range(len(runs_with_seeds))]
        return runs_with_seeds, expected_results
    else:
        return runs_with_seeds


def get_sweep_params(name: str, path: Path) -> list[dict] | tuple[list[dict], list[dict]]:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, "Could not find module"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sweep_params = module.SWEEP_PARAMS
    sweep_params = params_to_list(sweep_params)
    return sweep_params
