import influxdb_client
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from abc import ABC, abstractmethod

import datetime as dt
import numpy as np
import os
import gymnasium as gym
import pandas as pd
import time

from csv import DictReader
from math import floor
import asyncio

from typing import Callable
from corerl.utils.opc_connection import OpcConnection



class DBClientWrapper:
    def __init__(self, cfg, date_fn: Callable = None):
        self.bucket = cfg.bucket
        self.org = cfg.org
        self.client = influxdb_client.InfluxDBClient(url=cfg.url, token=cfg.token, org=self.org, timeout=30_000)
        self.write_client = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.start_time = np.inf
        self.end_time = -np.inf
        self.date_fn = date_fn

    def import_csv(self, root: str, date_col: str, col_names: list) -> None:
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

    def _parse_row(self, row: dict, date_col: str, col_names: list) -> Point:
        """
        Parse row of CSV file into Point 
        
        args:
            row: dict representing one row of csv
            date_col: label of date within the dict
            col_names: labels of data (i.e. non-date) columns in the dictionary
        """

        time = self.date_fn(row[date_col])
        time -= 7 * 60 * 60  # to adjust for the conversion between GMT and MST

        point = Point("reading")
        for i in range(len(col_names)):
            point = point.field(col_names[i], float(row[col_names[i]]))

        if self.start_time >= time:
            self.start_time = int(time)
        if self.end_time <= time:
            self.end_time = int(time)

        point = point.time(int(time * 1e9))  # multiplication to convert timestamp in seconds to nanoseconds
        return point

    def query(self, start_time: int, end_time: int, col_names: list | None = None,
              include_time: bool = False) -> pd.DataFrame:
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


class InfluxOPCEnv(ABC, gym.Env):
    def __init__(self, cfg):
        if cfg.db is not None:
            self.db_client = DBClientWrapper(cfg.db)
        else:
            self.db_client = None

        if cfg.opc is not None:
            self.opc_connection = OpcConnection(cfg.opc)
        else:
            self.opc_connection = None

        self.control_tags = cfg.control_tags
        self.col_names = cfg.col_names
        self.obs_length = cfg.obs_length

    async def _take_action(self,a: np.ndarray) -> None:
        await self.opc_connection.connect()
        nodes = await self.opc_connection.get_nodes(self.control_tags)
        variant_types = await self.opc_connection.read_variant_types(nodes)
        await self.opc_connection.write_values(nodes, variant_types, a)

    def take_action(self, a: np.ndarray):
        # NOTE: you may want to add handling if the action did not change, so running_take_action is not necessary
        asyncio.run(self._take_action(a))

    @abstractmethod
    def _get_reward(self, s: np.ndarray | pd.DataFrame, a: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _check_done(self) -> bool:
        raise NotImplementedError

    def get_observation(self, a: np.ndarray) -> (np.ndarray, float, bool, bool, dict):
        state = self._get_observation()
        done = self._check_done()
        reward = self._get_reward(state, a)
        state = state.to_numpy()
        return state, reward, done, False, {}

    def _get_observation(self) -> pd.DataFrame:
        now = floor(time.time())
        obs = self.db_client.query(now - self.obs_length, now, self.col_names)
        return obs

    def step(self, action: np.ndarray):
        self.take_action(action)
        end_timer = time.time() + self.obs_length
        time.sleep(end_timer - time.time())
        return self.get_observation(action)

    def reset(self) -> (np.ndarray, dict):
        state = self._get_observation()
        state = state.to_numpy()
        return state, {}

    def close(self):
        asyncio.run(self.opc_connection.disconnect())
