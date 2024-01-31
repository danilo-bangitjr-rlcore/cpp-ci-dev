import datetime as dt
import time
import numpy as np
import pandas as pd
from InfluxOPCEnv import InfluxOPCEnv, DBClientWrapperBase
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
        runtime, decision_freq=1800, observation_window=1800, last_n_observations=1700, offline_data_folder=None):
        super().__init__(db_client, opc_connection, control_tags, date_col, col_names, runtime, decision_freq, 
        observation_window, last_n_observations, offline_data_folder)
        self.control_tags_default = control_tag_default
        # TODO: figure these out
        self.observation_space = Box(low=np.ones(12)*-1000, high=np.ones(12)*1000) # What is this?
        self.action_space = Box(low=0, high=200)
        self.orp_sp = 100  # the ORP set point

    def set_orp_sp(self, sp):
        self.orp_sp = sp

    def _get_reward(self, s, a):
        mae = (s['ait301_pv'] - self.orp_sp).abs().mean()
        return  mae

    def reset(self, seed=0):
        self.take_action(self.control_tags_default)
        return super().reset(seed)
    
    def get_observation(self, a):
        self._update_now()
        self.state = self._get_observation() 
        done = self._check_done()
        reward = self._get_reward(self.state, a)
        return self.state.to_numpy(), reward, done, False, {}
