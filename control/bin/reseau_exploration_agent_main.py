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
    # Actual
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--init_fpm', default=15, type=int)
    parser.add_argument('--reset', default=True, type=bool)
    parser.add_argument('--reset_fpm', default=20, type=int)
    parser.add_argument('--reset_duration', default=3600, type=int) # 1 hour. Longer?
    parser.add_argument('--fpm_min', default=10, type=int)
    parser.add_argument('--fpm_max', default=50, type=int)
    parser.add_argument('--num_fpms', default=16, type=int)
    parser.add_argument('--orp_delay_start_times', default=['08:00:00', '13:00:00'], type=str, nargs='+')
    parser.add_argument('--orp_delay_duration', default=14400, type=int) # Duration of ORP Delay Experiment in seconds (14400 = 4 hours)
    # parser.add_argument('--orp_threshold_factor', default=2.0, type=float)

    # parser = argparse.ArgumentParser(description="run_file")
    # parser.add_argument('--init_fpm', default=10, type=int)
    # parser.add_argument('--reset', default=True, type=bool)
    # parser.add_argument('--reset_fpm', default=20, type=int)
    # parser.add_argument('--reset_duration', default=2, type=int) # 1 hour. Longer?
    # parser.add_argument('--fpm_min', default=10, type=int)
    # parser.add_argument('--fpm_max', default=50, type=int)
    # parser.add_argument('--num_fpms', default=3, type=int)
    # parser.add_argument('--orp_delay_start_times', default=['16:04:00', '16:05:00'], type=str, nargs='+')
    # parser.add_argument('--orp_delay_duration', default=10, type=int) # Duration of ORP Delay Experiment in seconds (14400 = 4 hours)
    # # parser.add_argument('--orp_threshold_factor', default=2.0, type=float)

    parser.add_argument('--decouple_steps',  default=0, type=int)
    parser.add_argument('--decision_freq', default=10, type=int) # frequency (s) for the agent to make decision
    parser.add_argument('--observation_window', default=10, type=int) # window (s) for getting observations
    
    cfg = parser.parse_args()

    cfg.env = env_factory.init_environment("Reseau", cfg)
    agent = agent_factory.init_agent("Reseau-Exploration", cfg)
    
    run_reseau_exploration_agent(agent, cfg)
