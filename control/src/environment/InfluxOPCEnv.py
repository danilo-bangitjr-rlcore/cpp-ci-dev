import influxdb_client
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

import datetime as dt
import numpy as np
import os
import time
import gymnasium as gym
import pandas as pd

from csv import DictReader
from math import floor
import asyncio
import time

class DBClientWrapperBase():
    def __init__(self, bucket, org, token, url, date_fn=None):
        self.bucket = bucket
        self.org = org
        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        self.write_client = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.start_time = np.inf
        self.end_time = -np.inf
        self.date_fn = date_fn
    
    def import_csv(self, root, date_col, col_names):
        self.col_names = col_names
        record = []
        for pth in os.listdir(root):
            if ".DS_Store" not in pth:
                full_pth = os.path.join(root, pth)
                dataset = DictReader(open(full_pth, 'r'))
                for datum in dataset:
                    point = self._parse_row(datum, date_col, col_names)
                    record.append(point)
        
        self.write_client.write(self.bucket, self.org, record)
        
        
    def _parse_row(self, row, date_col, col_names):
        """
        Parse row of CSV file into Point 
        
        args:
            row: dict representing one row of csv
            date_col: label of date within the dict
            col_names: labels of data (i.e. non-date) columns in the dictionary
          
        """
        
        time = self.date_fn(row[date_col])
        time -= 7*60*60 # to adjust for the conversion between GMT and MST     
        
        point = Point("reading")
        for i in range(len(col_names)):
            point = point.field(col_names[i], float(row[col_names[i]])) 
            
        if self.start_time >= time:
            self.start_time = int(time)
        if self.end_time <= time:
            self.end_time= int(time)
            
        point = point.time(int(time*1e9)) # multiplication to convert timestamp in seconds to nanoseconds
        return point


    def query(self, start_time, end_time, col_names=None, include_time=False):
        """
        Returns all data between start_time and end_time
        
        args:
            start_time (int) : a timestamp
            end_time (int) : a timestamp
            col_names (list) : list of columns to retreive. The default is to use self.col_names
        """
        
        assert end_time >= start_time
        
        if col_names is None:
            col_names = self.col_names
        
        if include_time:
            col_names = ["_time"] + col_names
        
        query_str_list = [
            'from(bucket:"{}") '.format(self.bucket),
            '|> range(start: {}, stop: {}) '.format(start_time, end_time),
            '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") '
            ]
        query_str = ' '.join(query_str_list)
        df_list = self.query_api.query_data_frame(query_str)
        
        if type(df_list) == list:
            df = pd.concat(df_list, axis=1)
        else:
            df = df_list
        df = df[col_names]
        return df
    
    
    
   
class InfluxOPCEnv(gym.Env):
    def __init__(self, db_client, opc_connection, control_tags, date_col, col_names, 
        runtime=None, decision_freq=10*60, observation_window=10, last_n_observations=None, offline_data_folder=None):
        # for continuing s
        self.db_client = db_client
        self.date_col = date_col,
        self.col_names = col_names
        
        if offline_data_folder is not None: # we will import data from a CSV
            assert date_col is not None
            assert col_names is not None
            self.db_client.import_csv(offline_data_folder, date_col, col_names)
            self.offline = True
        else:
            self.offline = False
            
        self.decision_freq = decision_freq
        self.observation_window = observation_window
        self.runtime = runtime
        self.opc_connection = opc_connection
        self.control_tags = control_tags
        self.last_n_observations = last_n_observations
        

    async def _take_action(self, a):
        await self.opc_connection.connect()
        # get the list of nodes
        # remember these are simulated nodes for these
        # write examples so we don't change real
        # values on the PLC
        nodes = await self.opc_connection.get_nodes(self.control_tags)
       
        # get the variant types
        # this is necessary to properly specify the
        # data types for the actual write operation
        # time.sleep(0.1)
        variant_types = await self.opc_connection.read_variant_types(nodes)
        
        # # write the values
        # # you need to provide 3 lists: the nodes, the variant types
        # # of the nodes, and the values. Of course, these should all be the same
        # # length
        await self.opc_connection.write_values(nodes, variant_types, a)


    def take_action(self, a):
        pass
        #asyncio.run(self._take_action(a))
      

    def _get_reward(self, s, a):
        raise NotImplementedError
    
    def _check_done(self):
        if self.state.size == 0 and self.offline:
            done = True
        elif self.runtime is not None: # not a continuing task
            if self._now>=self.start_time+self.runtime:
                done = True
            else: 
                done = False
        else:
            done = False
        return done
            
    
    def get_observation(self, a):
        """
        Takes a single synchronous environmental step. 
        """
        self._update_now()
        self.state = self._get_observation() 
        done = self._check_done()
        reward = self._get_reward(self.state, a)
        return self.state, reward, done, False, {}
        

    def _update_now(self):
        if self.offline:
            self._now += self.decision_freq
        else:
            self._now = floor(dt.datetime.timestamp(dt.datetime.now()))
    
    
    def _get_observation(self):
        """
        Gets observations, defined as all obvservations within the last self.decision_freq seconds from self._now
        
        returns: 
            self.state (pd.dataframe) : the observation
        """
        obs_df = self.db_client.query(self._now-self.observation_window, self._now, self.col_names)  
        obs = self.process_observation(obs_df)
        if self.last_n_observations is not None:
            obs=obs[-self.last_n_observations:] # only return the last n observations within a window. 
        return obs

    def reset(self, seed=0):
        """
        Resets the environment to its starting state

        Returns
        -------
        array_like of float
            The starting state feature representation
        """
        # ignores seed 

        self.start_time = self.db_client.start_time
        self._now = self.start_time
        self._update_now()
        self.state = self._get_observation()
        return self.state, {}
    
    
    def close(self):
        asyncio.run(self.opc_connection.disconnect())