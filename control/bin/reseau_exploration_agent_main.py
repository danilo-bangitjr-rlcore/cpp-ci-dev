import os, sys
sys.path.insert(0, '..')

import argparse
import src.environment.factory as env_factory
import src.agent.factory as agent_factory
import src.utils.utils as utils
from src.utils.run_reseau_exploration_agent import run_reseau_exploration_agent

os.chdir("..")
print("Change dir to", os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--init_fpm', default=200, type=int)
    parser.add_argument('--reset', default=True, type=bool)
    parser.add_argument('--reset_fpm', default=50, type=int) # Should it be 15?
    parser.add_argument('--reset_duration', default=1200, type=int) # 20 minutes
    parser.add_argument('--fpm_min', default=10, type=int)
    parser.add_argument('--fpm_max', default=200, type=int)
    parser.add_argument('--num_fpms', default=100, type=int)
    parser.add_argument('--orp_delay_start_times', default=['09:10:00', '13:10:00'], type=str, nargs='+')
    parser.add_argument('--orp_delay_duration', default=9600, type=int) # Duration of ORP Delay Experiment in seconds (9600 = 2 hours 40 minutes)
    parser.add_argument('--orp_threshold_factor', default=2.0, type=float)

    cfg = parser.parse_args()

    cfg.env = env_factory.init_environment("Reseau", cfg)
    agent = agent_factory.init_agent("Reseau-Exploration", cfg)
    
    run_reseau_exploration_agent(agent, cfg)
