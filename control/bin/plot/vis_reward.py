import os, sys
sys.path.insert(0, '../..')

import argparse
import src.environment.factory as env_factory
import src.utils.utils as utils
import src.utils.run_funcs as run_funcs

os.chdir("../..")
print("Change dir to", os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--version', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--env_name', default='NonContexTT', type=str)
    parser.add_argument('--env_info', default=[0., 3.], type=float, nargs='+') # go to the corresponding environment to check the specific setting
    parser.add_argument('--env_action_scaler', default=1., type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--discrete_control', default=0, type=int)
    parser.add_argument('--lr_constrain', default=1e-06, type=float)

    parser.add_argument('--clip_min', default=0, type=int)
    parser.add_argument('--clip_max', default=0, type=int)

    cfg = parser.parse_args()

    utils.set_seed(cfg.seed)
    train_env = env_factory.init_environment(cfg.env_name, cfg)

    title = "out/img/{}_{}_reward".format(cfg.env_name, "-".join([str(i) for i in cfg.env_info]))
    if cfg.clip_min != cfg.clip_max:
        clip = [cfg.clip_min, cfg.clip_max]
    else:
        clip = None
    run_funcs.vis_reward(train_env, title, clip)
