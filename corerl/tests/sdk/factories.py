"""
Factory classes for creating test data objects.

This module provides factory classes that simplify the creation of complex test data,
making tests more readable and maintainable.
"""

import datetime
from typing import Any

import numpy as np
import pandas as pd
from lib_agent.buffer.datatypes import DataMode

from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, TemporalState


class PipelineFrameFactory:
    def __init__(self):
        self._default_data_mode = DataMode.ONLINE
        self._default_timestamp_start = datetime.datetime.now(datetime.UTC)
        self._default_timestamp_delta = datetime.timedelta(minutes=5)

    def create(
        self,
        data: pd.DataFrame | dict[str, Any] | None = None,
        data_mode: DataMode | None = None,
        last_stage: StageCode | None = None,
        temporal_state: TemporalState | None = None,
        n_rows: int = 4,
        timestamp_start: datetime.datetime | None = None,
        timestamp_delta: datetime.timedelta | None = None,
        tag_names: list[str] | None = None,
        tag_values: dict[str, list] | None = None,
        index: pd.DatetimeIndex | list[datetime.datetime] | None = None,
    ) -> PipelineFrame:
        """
        Create a PipelineFrame with specified or default data.
        """
        data_mode = data_mode or self._default_data_mode
        temporal_state = temporal_state or {}
        timestamp_start = timestamp_start or self._default_timestamp_start
        timestamp_delta = timestamp_delta or self._default_timestamp_delta

        if data is None:
            data = self._create_sample_data(
                n_rows=n_rows,
                timestamp_start=timestamp_start,
                timestamp_delta=timestamp_delta,
                tag_names=tag_names,
                tag_values=tag_values,
            )
        elif isinstance(data, dict):
            if index is not None:
                if isinstance(index, list):
                    index = pd.DatetimeIndex(index)
                data = pd.DataFrame(data, index=index)
            else:
                timestamps = [timestamp_start + i * timestamp_delta for i in range(len(next(iter(data.values()))))]
                data = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))

        pf = PipelineFrame(data=data, data_mode=data_mode, last_stage=last_stage)
        pf.temporal_state = temporal_state

        return pf

    def create_with_tags(
        self,
        tag_configs: dict[str, dict[str, Any]],
        n_rows: int = 4,
        data_mode: DataMode | None = None,
        last_stage: StageCode | None = None,
        temporal_state: TemporalState | None = None,
        timestamp_start: datetime.datetime | None = None,
        timestamp_delta: datetime.timedelta | None = None,
    ) -> PipelineFrame:
        """
        Create a PipelineFrame with specific tag configurations.

        Example:
            >>> factory = PipelineFrameFactory()
            >>> pf = factory.create_with_tags({
            ...     'sensor_1': {'values': [1, 2, 3, 4], 'type': 'sensor'},
            ...     'action_1': {'values': [0, 0, 1, 1], 'type': 'ai_setpoint'},
            ...     'reward': {'values': [0.1, 0.2, 0.3, 0.4], 'type': 'meta'},
            ... })
        """
        data = {}
        for tag_name, config in tag_configs.items():
            if "values" in config:
                data[tag_name] = config["values"]
            else:
                tag_type = config.get("type", "sensor")
                data[tag_name] = self._generate_values_for_tag_type(tag_type, n_rows)

        return self.create(
            data=data,
            data_mode=data_mode,
            last_stage=last_stage,
            temporal_state=temporal_state,
            timestamp_start=timestamp_start,
            timestamp_delta=timestamp_delta,
        )

    def create_simple(
        self,
        sensor_tags: int = 2,
        action_tags: int = 1,
        meta_tags: list[str] | None = None,
        n_rows: int = 4,
        data_mode: DataMode | None = None,
        last_stage: StageCode | None = None,
        temporal_state: TemporalState | None = None,
        timestamp_start: datetime.datetime | None = None,
        timestamp_delta: datetime.timedelta | None = None,
    ) -> PipelineFrame:
        """
        Create a simple PipelineFrame with common tag patterns.
        """
        meta_tags = meta_tags or ["reward"]

        data = {}

        for i in range(sensor_tags):
            data[f"sensor_{i + 1}"] = np.random.uniform(0, 10, n_rows).tolist()

        for i in range(action_tags):
            data[f"action_{i + 1}"] = np.random.uniform(-1, 1, n_rows).tolist()

        for tag_name in meta_tags:
            if tag_name == "reward":
                data[tag_name] = np.random.uniform(0, 1, n_rows).tolist()
            elif tag_name in ["terminated", "truncated"]:
                data[tag_name] = [False] * (n_rows - 1) + [True]
            else:
                data[tag_name] = np.random.uniform(0, 1, n_rows).tolist()

        return self.create(
            data=data,
            data_mode=data_mode,
            last_stage=last_stage,
            temporal_state=temporal_state,
            timestamp_start=timestamp_start,
            timestamp_delta=timestamp_delta,
        )

    def _create_sample_data(
        self,
        n_rows: int,
        timestamp_start: datetime.datetime,
        timestamp_delta: datetime.timedelta,
        tag_names: list[str] | None,
        tag_values: dict[str, list] | None,
    ) -> pd.DataFrame:
        """Create sample DataFrame with default structure."""
        tag_names = tag_names or ["sensor_1", "sensor_2", "action_1", "reward"]
        tag_values = tag_values or {}

        timestamps = [timestamp_start + i * timestamp_delta for i in range(n_rows)]

        data = {}
        for tag_name in tag_names:
            if tag_name in tag_values:
                data[tag_name] = tag_values[tag_name]
            elif "sensor" in tag_name.lower():
                data[tag_name] = np.random.uniform(0, 10, n_rows).tolist()
            elif "action" in tag_name.lower():
                data[tag_name] = np.random.uniform(-1, 1, n_rows).tolist()
            elif tag_name == "reward":
                data[tag_name] = np.random.uniform(0, 1, n_rows).tolist()
            elif tag_name in ["terminated", "truncated"]:
                data[tag_name] = [False] * (n_rows - 1) + [True]
            else:
                data[tag_name] = np.random.uniform(0, 1, n_rows).tolist()

        return pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))

    def _generate_values_for_tag_type(self, tag_type: str, n_rows: int) -> list:
        """Generate appropriate values based on tag type."""
        if tag_type == "ai_setpoint":
            return np.random.uniform(-1, 1, n_rows).tolist()
        if tag_type == "meta":
            return np.random.uniform(0, 1, n_rows).tolist()
        if tag_type == "sensor":
            return np.random.uniform(0, 10, n_rows).tolist()
        return np.random.uniform(0, 1, n_rows).tolist()

    @staticmethod
    def build(
        data: pd.DataFrame | dict[str, Any] | None = None,
        data_mode: DataMode | None = None,
        last_stage: StageCode | None = None,
        temporal_state: TemporalState | None = None,
        n_rows: int = 4,
        timestamp_start: datetime.datetime | None = None,
        timestamp_delta: datetime.timedelta | None = None,
        tag_names: list[str] | None = None,
        tag_values: dict[str, list] | None = None,
        index: pd.DatetimeIndex | list[datetime.datetime] | None = None,
    ) -> PipelineFrame:
        """
        Convenience method to create a PipelineFrame in one call.

        Creates a factory instance and calls create() with the provided arguments.
        Perfect for tests that need a single PipelineFrame without repeated factory instantiation.
        """
        factory = PipelineFrameFactory()
        return factory.create(
            data=data,
            data_mode=data_mode,
            last_stage=last_stage,
            temporal_state=temporal_state,
            n_rows=n_rows,
            timestamp_start=timestamp_start,
            timestamp_delta=timestamp_delta,
            tag_names=tag_names,
            tag_values=tag_values,
            index=index,
        )
