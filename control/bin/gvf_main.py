import os, sys
sys.path.insert(0, '..')

import argparse
import src.environment.factory as env_factory
import src.agent.factory as agent_factory
import src.utils.utils as utils
import src.utils.run_gvf as run_gvf
from src.prediction.gvf import GVF

os.chdir("..")
print("Change dir to", os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    # Experiment
    parser.add_argument('--exp_name', default='gvf_offline_csv', type=str)
    parser.add_argument('--exp_info', default='', type=str)
    parser.add_argument('--param', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cpu', type=str)

    # GVF
    parser.add_argument('--gvf', default='FC', type=str)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--layer_norm', default=0, type=int)
    parser.add_argument('--layer_init', default='Xavier', type=str)
    parser.add_argument('--activation', default='ReLU6', type=str)
    parser.add_argument('--actor_optimizer', default='RMSprop', type=str)
    parser.add_argument('--obs_normalizer', default='Identity', type=str)
    parser.add_argument('--obs_scale', default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--obs_bias', default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], type=float, nargs='+')
    parser.add_argument('--cumulant_normalizer', default='Identity', type=str)
    parser.add_argument('--cumulant_scale', default=1.0, type=float)
    parser.add_argument('--cumulant_bias', default=0.0, type=float)
    parser.add_argument('--buffer_size', default=1, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--use_target_network', default=1, type=int)
    parser.add_argument('--polyak', default=0.995, type=float) # 0 is hard sync
    parser.add_argument('--hidden_gvf', default=[256, 256], type=int, nargs='+')
    parser.add_argument('--lr_gvf', default=0.001, type=float)

    # Offline Training
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--offline_src', default='CSV', type=str)
    parser.add_argument('--offline_logs_dir', default='Reseau_WTP_Data', type=str)
    parser.add_argument('--offline_log_files', default=['datalog_2023_12.csv', 'datalog_2024_1.csv'], type=str, nargs='+')
    parser.add_argument('--offline_log_tags', default=["Date", "AIT101", "AIT301", "AIT401", "FIT101", "FIT210", "FIT230", "FIT250", "FIT401", "PT100", "PT101", "PT151", "PT161", "PT171", "PT301", "PT302"], type=str, nargs='+')
    parser.add_argument('--tags_to_drop', default=["AIT301", "AIT401", "PT151", "PT301"], type=str, nargs='+')
    parser.add_argument('--cumulant_tag', default="AIT301", type=str)
    parser.add_argument('--data_freq', default=60, type=int) # In seconds
    parser.add_argument('--time_step_freq', default=600, type=int) # In seconds
    parser.add_argument('--train_split', default=0.8, type=float)
    parser.add_argument('--validation_split', default=0.1, type=float)
    parser.add_argument('--test_split', default=0.1, type=float)

    cfg = parser.parse_args()

    cfg.offline_logs_path = './data/{}/'.format(cfg.offline_logs_dir)
    cfg.exp_path = './out/output/{}/param_{}/seed_{}/'.format(cfg.exp_name, cfg.param, cfg.seed)  # savelocation for logs
    cfg.parameters_path = os.path.join(cfg.exp_path, "parameters")
    utils.ensure_dir(cfg.exp_path)

    utils.set_seed(cfg.seed)

    utils.write_json(cfg.exp_path, cfg) # write json after finishing all parameter changing.

    gvf = GVF(cfg)

    run_gvf.train_val_test(gvf, cfg)
