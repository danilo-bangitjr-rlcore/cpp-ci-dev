import subprocess
import os, sys
sys.path.insert(0, '..')
import datetime
import json
from types import SimpleNamespace
import argparse
import src.environment.factory as env_factory
import src.agent.factory as agent_factory
import src.utils.utils as utils
import src.utils.run_funcs as run_funcs

os.chdir("..")
print("Change dir to", os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--load_from_json', default='', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--version', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--param', default=0, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--exp_name', default='temp', type=str)
    parser.add_argument('--exp_info', default='', type=str)
    parser.add_argument('--timeout', default=1, type=int)
    parser.add_argument('--max_steps', default=20000, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--log_test', default=0, type=int)
    parser.add_argument('--stats_queue_size', default=1, type=int)
    parser.add_argument('--evaluation_criteria', default='return', type=str)
    parser.add_argument('--render', default=0, type=int)

    parser.add_argument('--env_name', default='ThreeTank', type=str)
    parser.add_argument('--env_info', default=[0., 3.], type=float, nargs='+') # go to the corresponding environment to check the specific setting
    parser.add_argument('--env_action_scaler', default=1., type=float)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--discrete_control', default=0, type=int)

    parser.add_argument('--agent_name', default='SimpleAC', type=str)
    parser.add_argument('--actor', default='Beta', type=str)
    parser.add_argument('--critic', default='FC', type=str)
    parser.add_argument('--layer_norm', default=0, type=int)
    parser.add_argument('--layer_init_actor', default='Xavier/1', type=str)
    parser.add_argument('--layer_init_critic', default='Xavier/1', type=str)
    parser.add_argument('--activation', default='ReLU', type=str)
    parser.add_argument('--head_activation', default='Softplus', type=str)
    parser.add_argument('--optimizer', default='RMSprop', type=str)
    parser.add_argument('--state_normalizer', default='Identity', type=str)
    parser.add_argument('--action_normalizer', default='Identity', type=str)
    parser.add_argument('--reward_normalizer', default='Identity', type=str)
    parser.add_argument('--exploration', default=0.1, type=float)
    parser.add_argument('--beta_parameter_bias', default=0., type=float)
    parser.add_argument('--beta_parameter_bound', default=1e8, type=float)
    parser.add_argument('--action_scale', default=1., type=float)
    parser.add_argument('--action_bias', default=0., type=float)
    parser.add_argument('--auto_calibrate_beta_support', default=0, type=int)
    
    parser.add_argument('--load_path', default="", type=str)
    parser.add_argument('--load_checkpoint', default=1, type=int)
    parser.add_argument('--buffer_size', default=1, type=int)
    parser.add_argument('--buffer_prefill', default=0, type=int)
    parser.add_argument('--etc_buffer_prefill', default=0, type=int) # buffer prefill for etc 
    parser.add_argument('--etc_learning_start', default=0, type=int) # buffer prefill for etc 
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--use_target_network', default=1, type=int)
    parser.add_argument('--polyak', default=0, type=float) # 0 is hard sync
    parser.add_argument('--hidden_actor', default=[256, 256], type=int, nargs='+')
    parser.add_argument('--hidden_critic', default=[256, 256], type=int, nargs='+')
    parser.add_argument('--lr_actor', default=0.0001, type=float)
    parser.add_argument('--lr_critic', default=0.001, type=float)
    parser.add_argument('--lr_v', default=0.001, type=float)
    parser.add_argument('--lr_constrain', default=1e-06, type=float)

    parser.add_argument('--tau', default=0.000001, type=float)
    parser.add_argument('--rho', default=0.1, type=float)
    parser.add_argument('--prop_rho_mult', default=2.0, type=float)
    parser.add_argument('--n', default=30, type=int)

    # Explore Then Commit
    parser.add_argument('--actions_per_dim', default=50, type=int)
    parser.add_argument('--min_trials', default=1, type=int)

    cfg = parser.parse_args()

    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    cfg.githash = sha

    if cfg.load_from_json == '':
        cfg.exp_path = './out/output/test_v{}/{}/{}{}/{}/param_{}/seed_{}/'.format(cfg.version, cfg.env_name, cfg.exp_name, cfg.exp_info, cfg.agent_name, cfg.param, cfg.seed)  # savelocation for logs
    else:
        jf = cfg.load_from_json
        assert os.path.isfile(jf), print("JSON FILE DOES NOT EXIST")
        with open(jf, 'r') as f:
            jcfg = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        for k in cfg.__dict__.keys():
            if getattr(jcfg, k, None) is None:
                print("Adding missing key: {}={}".format(k, getattr(cfg, k)))
                setattr(jcfg, k, getattr(cfg, k))
        cfg = jcfg
        ts = datetime.datetime.now().timestamp()
        cfg.exp_path = cfg.exp_path[:-1] + "_{}".format(ts)
    cfg.parameters_path = os.path.join(cfg.exp_path, "parameters")
    cfg.vis_path = os.path.join(cfg.exp_path, "visualizations")
    utils.ensure_dir(cfg.exp_path)
    utils.ensure_dir(cfg.parameters_path)
    utils.ensure_dir(cfg.vis_path)

    utils.set_seed(cfg.seed)
    cfg.train_env = env_factory.init_environment(cfg.env_name, cfg)
    cfg.eval_env = env_factory.init_environment(cfg.env_name, cfg)
    
    if not cfg.discrete_control:
        env_factory.configure_action_scaler_and_bias(cfg)

    utils.write_json(cfg.exp_path, cfg) # write json after finishing all parameter changing.
    cfg.logger = utils.logger_setup(cfg)
    agent = agent_factory.init_agent(cfg.agent_name, cfg)
    run_funcs.run_steps(agent, cfg.max_steps, cfg.log_interval, cfg.log_test, cfg.exp_path, cfg.buffer_prefill)
