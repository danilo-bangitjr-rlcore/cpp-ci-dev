import itertools
import importlib.util
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import pickle as pkl


def flatten_list(nd_list: list) -> list:
    flat_list = []

    def flatten(item):
        if isinstance(item, list):
            for sub_item in item:
                flatten(sub_item)
        else:
            flat_list.append(item)

    flatten(nd_list)
    return flat_list


def params_to_list(params: dict) -> list[dict]:
    keys, values = zip(*params['independent'].items())
    runs_ = [dict(zip(keys, v)) for v in itertools.product(*values)]

    runs = []
    for run in runs_:
        for cond_key, cond_fn in params['conditional'].items():
            cond_value = cond_fn(run)
            if cond_value is not None:
                run[cond_key] = cond_value

        for k in run.keys():
            if not isinstance(run[k], list):
                run[k] = [run[k]]

        keys, values = zip(*run.items())
        run = [dict(zip(keys, v)) for v in itertools.product(*values)]
        runs.append(run)

    runs = flatten_list(runs)
    for i in range(len(runs)):
        runs[i]['experiment.param'] = i

    return flatten_list(runs)


def get_sweep_params(name: str, path: Path) -> list[dict]:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, "Could not find module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sweep_params = module.SWEEP_PARAMS
    sweep_params = params_to_list(sweep_params)
    return sweep_params


def get_nested_value(d: dict, path: str) -> object:
    keys = path.split('.')
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None  # or raise an error
    return current


def get_sweep_results(path: Path, config_keys: list[str], steps: list[int], step_keys: list[str]) -> list[dict]:
    config_list = []
    for p in path.rglob("*"):
        if 'seed' in p.name.split("/")[-1]:
            # first get config info
            return_dict = {}
            config = OmegaConf.load(p / 'config.yaml')
            for key in config_keys:
                return_dict[key] = OmegaConf.select(config, key)

            # next, retrieve step log info
            return_dict['step_logs'] = {}
            for step_log_path in (p / 'logs').iterdir():
                s = step_log_path.name
                step = int(s[s.find('-') + len('-'):s.rfind('.pkl')])  # https://stackoverflow.com/a/18790509
                if step in steps:
                    step_log_ = pkl.load(open(step_log_path, 'rb'))
                    step_log = {}
                    for key in step_keys:
                        step_log[key] = get_nested_value(step_log_, key)
                    return_dict['step_logs'][step] = step_log

            config_list.append(return_dict)

    return config_list


def list_to_df(lst: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(lst)
