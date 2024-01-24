import copy
import os, sys
import json
import logging
import numpy as np
import torch


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
