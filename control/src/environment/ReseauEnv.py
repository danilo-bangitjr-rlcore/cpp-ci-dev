import datetime as dt
import time
import numpy as np
from src.environment.InfluxOPCEnv import InfluxOPCEnv
from gymnasium.spaces import Box

def date_to_timestamp_reseau(date):
    """
    Input: date string 
    Returns: timestamp for that date
    """
    
    # add leading 0 to hour
    time_list = date.split(' ')
    if time_list[1][1] == ':':
        time_list[1] = "0" + time_list[1]
    time_str = " ".join(time_list)
    
    formats = ['%Y-%m-%d %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p'] #valid formats in the dataset
    valid_format = False

    for format in formats:
        try:
            time = dt.datetime.strptime(time_str, format)
            valid_format   = True
            break
        except:
            pass
    assert valid_format
    return dt.datetime.timestamp(time)


class ReseauEnv(InfluxOPCEnv):
    def __init__(self, db_client, opc_connection, control_tags, control_tag_default, date_col, col_names, 
        runtime, decision_freq=10 * 60, observation_window=10, offline_data_folder=None):
        super().__init__(db_client, opc_connection, control_tags, date_col, col_names, runtime, decision_freq, observation_window, offline_data_folder)
        self.control_tags_default = control_tag_default
        # TODO: figure these out
        self.observation_space = Box(low=np.ones(12)*-1000, high=np.ones(12)*1000) # What is this?
        self.action_space = Box(low=0, high=200)

    def _get_reward(self, s, a):
        return  0
    
    def process_observation(self, obs):
        return np.mean(obs.to_numpy(), axis=0) # averages over timesteps

    def reset(self, seed=0):
        self.take_action(self.control_tags_default)
        time.sleep(0.1)
        return super().reset(seed)