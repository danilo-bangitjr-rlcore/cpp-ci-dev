import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from abc import ABC, abstractmethod

from corerl.utils.hook import when, Hooks

import numpy as np
import gymnasium as gym
import pandas as pd
from datetime import datetime, timedelta, UTC
from typing import Tuple, Generator, List
import asyncio
import random

from typing import Callable, List
from corerl.utils.opc_connection import OpcConnection
from corerl.sql_logging.sql_logging import get_sql_engine
import logging

logger = logging.getLogger(__name__)


class DBClientWrapper:
    def __init__(self, cfg):
        logger.debug(f"{cfg=}")
        self.sensor_table = cfg.sensor_table
        self.db_con = get_sql_engine(db_data=cfg, db_name=cfg.sensor_db_name)

        # influx setup
        self.bucket = cfg.influx.bucket
        org = cfg.influx.org
        self.client = influxdb_client.InfluxDBClient(
            url=cfg.influx.url, token=cfg.influx.token, org=org, timeout=30_000
        )
        self.write_client = self.client.write_api(write_options=SYNCHRONOUS)
        self.influx_query_api = self.client.query_api()

    def _influx_query(
        self,
        start_time: datetime,
        end_time: datetime,
        col_names: List[str],
    ) -> pd.DataFrame:
        filter_list = [f'r._field == "{field_name}"' for field_name in col_names]
        filter_str = " or\n".join(filter_list)
        timescale = f"10s"
        start_t_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_t_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        query_str = f"""from(bucket: "{self.bucket}")
          |> range(start: {start_t_str}, stop: {end_t_str})
          |> filter(fn: (r) => {filter_str})
          |> aggregateWindow(every: {timescale}, fn: mean, createEmpty: false)
          |> drop(columns: ["result", "id", "table", "_start", "_stop", "_measurement", "host"])
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")"""
        # query_str = " ".join(query_str_list)
        df_list = self.influx_query_api.query_data_frame(query_str)
        if type(df_list) == list:
            df = pd.concat(df_list, axis=1)
        else:
            df = df_list

        return df

    def timescale_get_last(self, end_time: datetime, col_names: List[str]) -> pd.Series:
        end_t_str = end_time.isoformat()

        dfs = []
        for col_name in col_names:
            query_str = f"""
                SELECT
                time,
                name,
                (fields->'val')::float AS val
                FROM {self.sensor_table}
                WHERE name = '{col_name}'
                AND time < TIMESTAMP '{end_t_str}'
                ORDER BY time DESC
                LIMIT 1;
            """
            sensor_data = pd.read_sql(sql=query_str, con=self.db_con)
            sensor_data["time"] = end_time

            if sensor_data.empty:
                logger.warning(f"failed query:\n{query_str}")
                raise Exception("dataframe returned from timescale was empty.")

            sensor_data = sensor_data.pivot(columns="name", values="val", index="time")
            dfs.append(sensor_data)

        df = pd.concat(dfs, ignore_index=False, axis=1)
        logger.debug(f"timescale_get_last df\n{df}")
        missing_cols = set(col_names) - set(df.columns)
        df[list(missing_cols)] = np.nan

        df = df[col_names]
        df = df.iloc[-1]  # gets last row of df as series

        return df

    def _timescale_query(
        self,
        start_time: datetime,
        end_time: datetime,
        col_names: List[str],
    ) -> pd.DataFrame:
        start_t_str = start_time.isoformat()
        end_t_str = end_time.isoformat()

        name_filter = "\tOR ".join([f"name = '{col}'\n" for col in col_names])
        name_filter = f"({name_filter})"
        query_str = f"""
            SELECT 
              name,
              avg((fields->'val')::float) AS avg_val
            FROM {self.sensor_table}
            WHERE time > TIMESTAMP '{start_t_str}'
            AND time < TIMESTAMP '{end_t_str}'
            AND {name_filter}
            GROUP BY name
            ORDER BY name ASC;
        """

        sensor_data = pd.read_sql(sql=query_str, con=self.db_con)
        if sensor_data.empty:
            logger.warning(f"failed query:\n{query_str}")
            raise Exception("dataframe returned from timescale was empty.")
        sensor_data["time"] = end_time

        sensor_data = sensor_data.pivot(columns="name", values="avg_val", index="time")

        return sensor_data

    def query(
        self,
        start_time: datetime,
        end_time: datetime,
        col_names: List[str],
        include_time: bool = False,
    ) -> pd.Series:
        """
        Returns all data between start_time and end_time

        args:
            start_time (int) : a timestamp
            end_time (int) : a timestamp
            col_names (list) : list of columns to retreive. The default is to use self.col_names
        """

        assert end_time >= start_time

        if include_time:
            col_names = ["_time"] + col_names

        try:
            df = self._timescale_query(start_time=start_time, end_time=end_time, col_names=col_names)
        except:
            logger.warning(f"timescale query failed.", exc_info=True)
            df = self._influx_query(start_time=start_time, end_time=end_time, col_names=col_names)

        missing_cols = set(col_names) - set(df.columns)
        df[list(missing_cols)] = np.nan

        df = df[col_names]
        df = df.iloc[-1]  # gets last row of df as series

        return df


def generate_times(ts: datetime, interval: timedelta):
    while True:
        ts += interval
        yield ts


class InfluxOPCEnv(ABC, gym.Env):
    clock: Generator

    def __init__(self, cfg):
        self._hooks = Hooks(keys=[e.value for e in when.Env])
        self.db_client = DBClientWrapper(cfg.db)

        self.opc_connection = OpcConnection(cfg.opc)

        self.control_tags = cfg.control_tags
        self.col_names = cfg.obs_col_names
        self.obs_length = timedelta(seconds=cfg.obs_length)
        self.prev_action_ts = None
        self.needs_warmup: bool = False
        self.constant_sp = cfg.constant_sp
        self.setpoints_to_write = {}
        self.heartbeat_interval = cfg.heartbeat_interval
        self.telegraf_collection_interval = cfg.telegraf_collection_interval
        self.telegraf_flush_interval = cfg.telegraf_flush_interval
        self.reset_clock()

    async def setup_setpoint_to_write(self) -> None:
        setpoints_to_write = {}

        constant_sp_addresses = list(self.constant_sp.keys())
        constant_sp_values = [self.constant_sp[k] for k in constant_sp_addresses]

        control_sp_addresses = list(self.control_tags)
        control_sp_values = await self.get_deployed_action()

        sp_addresses = [*constant_sp_addresses, *control_sp_addresses]
        sp_values = [*constant_sp_values, *control_sp_values.tolist()]
        for key, val in zip(sp_addresses, sp_values):
            setpoints_to_write[key] = val

        logger.debug(f"{sp_addresses=}")
        logger.debug(f"{sp_values=}")
        self.setpoints_to_write = setpoints_to_write

    def reset_clock(self) -> None:
        self.clock = generate_times(ts=datetime.now(tz=UTC), interval=self.obs_length)

    @abstractmethod
    async def healthcheck(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def async_reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        raise NotImplementedError

    async def check_opc_client(self) -> None:
        try:
            logger.debug("Checking OPC connection...")
            await self.opc_connection.client.check_connection()
        except Exception:
            logger.warning(f"OPC connection failed", exc_info=True)
            await self.attempt_reconnect()

    async def attempt_reconnect(self) -> None:
        logger.info(f"Attempting to reconnect to OPC")
        try:
            if hasattr(self.opc_connection, "client"):
                await self.opc_connection.client.disconnect()

            await self.opc_connection.connect()
            logger.info("OPC reconnection successful")
            self.signal_warmup()

        except:
            logger.warning("OPC reconnection failed.")

    async def initialize_connection(self) -> None:
        await self.opc_connection.connect()

    async def _take_action(self, a: np.ndarray, addresses: list[str] | None = None) -> None:
        if addresses is None:
            addresses = list(self.control_tags)
        assert isinstance(addresses, list)
        nodes = await self.opc_connection.get_nodes(addresses)
        variant_types = await self.opc_connection.read_variant_types(nodes)
        await self.opc_connection.write_values(nodes, variant_types, a)
        self.prev_action_ts = datetime.now()

    def take_action(self, a: np.ndarray):
        # NOTE: you may want to add handling if the action did not change, so running_take_action is not necessary
        asyncio.run(self._take_action(a))

    @abstractmethod
    def _get_reward(
        self,
        obs: np.ndarray,
        obs_series: pd.Series,
        steps_until_decision: int | None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def _check_done(self) -> bool:
        raise NotImplementedError

    def get_observation(
        self,
        a: np.ndarray,
        stop_t: datetime,
        decision_point: bool,
        steps_until_decision: int | None,
        info: dict = {},
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        state = self._get_observation(stop_t)
        done = self._check_done()
        logger.debug("compute reward")
        reward = self._get_reward(
            obs=state.to_numpy(),
            action=a,
            obs_series=state,
            decision_point=decision_point,
            steps_until_decision=steps_until_decision,
        )
        state = state.to_numpy().squeeze()
        logger.debug(f"{self.col_names=}")
        logger.debug(f"raw_obs={state}")
        return state, reward, done, False, info

    def _get_observation(self, stop_t: datetime) -> pd.Series:
        logger.debug("_get_observation")
        obs = self.db_client.query(
            start_time=stop_t - self.obs_length,
            end_time=stop_t,
            col_names=self.col_names,
        )
        logger.debug("_get_observation successful")
        return obs

    def step(self, action: np.ndarray):
        raise NotImplementedError("Only supports async_step")

    @abstractmethod
    async def get_deployed_action(self, time: datetime | None = None) -> np.ndarray:
        raise NotImplementedError

    async def _async_step(
        self,
        action: np.ndarray,
        decision_point: bool,
        steps_until_decision: int | None = None,
        time: datetime | None = None,
    ):
        if decision_point:
            logger.info("*** Decision Point: writing action to OPC ***")
            await self._take_action(action)
            logger.info(f"Writing constant setpoints: {self.constant_sp}")
            addresses = list(self.constant_sp.keys())
            constant_values = [self.constant_sp[k] for k in addresses]
            # add jitter
            constant_values = [val + 2e-4 * (random.random() - 0.5) for val in constant_values]
            await self._take_action(a=np.array(constant_values), addresses=addresses)
            # grace period to allow new action to take effect
            grace_period = (
                self.telegraf_collection_interval + self.telegraf_flush_interval + self.heartbeat_interval + 2
            )
            if grace_period > self.obs_length.total_seconds():
                logger.warning(
                    f"post _take_action grace period {grace_period} longer than self.obs_length: {self.obs_length}"
                )
            await asyncio.sleep(grace_period)

        deployed_action = await self.get_deployed_action()
        info = {"deployed_action": deployed_action}
        if time is None:
            stop_t = next(self.clock)
        else:
            stop_t = time

        assert stop_t.tzinfo == UTC

        now = datetime.now(tz=UTC)
        if stop_t >= now:
            wait_duration = stop_t - now
            await asyncio.sleep(wait_duration.total_seconds())
        else:
            logger.warning("step called after nominal wait period, resetting step clock...")
            self.reset_clock()
            logger.info("Using current timestamp to generate next observation")
            stop_t = now
        # TODO: make influx query async
        # for now, get_observation blocks

        s, r, term, trunc, info = self.get_observation(
            action,
            stop_t,
            decision_point,
            steps_until_decision,
            info,
        )

        _kwargs = self.get_all_data()

        self._hooks(
            when.Env.AfterStep,
            self,
            None,
            action,
            r,
            s,
            term,
            trunc,
            **_kwargs,
        )

        return s, r, term, trunc, info

    async def async_step(
        self,
        action: np.ndarray,
        decision_point: bool,
        steps_until_decision: int | None = None,
        time: datetime | None = None,
    ):
        """
        TODO:
        If env step fails, we should update the state before trying again
        This requires either:
            (1) fudging the traces in the state construction, or
            (2) resetting the state constructor and warming it up with recent data
        For now we accept that the action may be outdated.
        """
        while True:
            try:
                return await self._async_step(action, decision_point, steps_until_decision, time)
            # except Exception as e:
            except Exception:
                logger.warning(f"async env step failed!", exc_info=True)
                await self.check_opc_client()
                await asyncio.sleep(10)

    def signal_warmup(self):
        logger.info("Signaling need for new warmup")
        self.needs_warmup = True

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.options = options
        now = datetime.now(tz=UTC)
        state = self._get_observation(stop_t=now)
        state = state.to_numpy().squeeze()
        return state, {}

    def close(self):
        asyncio.run(self.opc_connection.disconnect())

    def get_all_data(self):
        return {}

    def register_hook(self, hook, when: when.Env):
        self._hooks.register(hook, when)
