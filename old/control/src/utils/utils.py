import copy
import os, sys
import json
import logging
import numpy as np
import torch
import pandas as pd


class Logger:
    def __init__(self, config, log_dir):
        log_file = os.path.join(log_dir, 'log')
        self._logger = logging.getLogger()
        
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        
        self._logger.setLevel(level=logging.INFO)
        
        self.config = config
        # if config.tensorboard_logs: self.tensorboard_writer = SummaryWriter(config.get_log_dir())
    
    def info(self, log_msg):
        self._logger.info(log_msg)


def ensure_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)
    return

def write_json(log_dir, cfg):
    write_down = copy.deepcopy(cfg.__dict__)
    write_down.pop("train_env")
    write_down.pop("eval_env")
    with open('{}/config.json'.format(log_dir), 'w') as f:
        pretty_json = json.dumps(write_down, indent=4)
        f.write(pretty_json)

def logger_setup(cfg):
    logger = Logger(cfg, cfg.exp_path)
    return logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# This currently assumes that each row should be treated as a (s, c) pair but we need to take into account the time step
# Two pointer approach? First pointer points at s and second pointer points at s'?
def load_offline_logs(cfg):
    dfs = []
    for filename in cfg.offline_log_files:
        df = pd.read_csv(cfg.offline_logs_path + filename, names=cfg.offline_log_tags, header=0)
        # dfs.append(df[1:])

    all_data = pd.concat(dfs, ignore_index=True)

    dates = pd.to_datetime(all_data["Date"])
    all_data = all_data.drop(['Date'], axis=1)

    all_data = all_data.astype(np.float32)

    cumulants = all_data[cfg.cumulant_tag]
    all_data = all_data.drop([cfg.cumulant_tag], axis=1)

    #actions = all_data[cfg.action_tag]
    #all_data = all_data.drop([cfg.action_tag], axis=1)

    observations = all_data.drop(cfg.tags_to_drop, axis=1)

    return dates.to_numpy(), observations.to_numpy(), cumulants.to_numpy()
