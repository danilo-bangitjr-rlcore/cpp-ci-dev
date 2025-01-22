import logging
from dataclasses import field

import pandas as pd

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.eval.base_eval import BaseEvalConfig
from corerl.state import AppState


def mean_by_tag(df: pd.DataFrame, tag: str) -> float:
    return float(df[tag].dropna().mean())


def variance_by_tag(df: pd.DataFrame, tag: str) -> float:
    return float(df[tag].dropna().var()) # type: ignore


def max_by_tag(df: pd.DataFrame, tag: str) -> float:
    return float(df[tag].dropna().max())


def min_by_tag(df: pd.DataFrame, tag: str) -> float:
    return float(df[tag].dropna().min())


def percentile_by_tag(df: pd.DataFrame, tag: str, percentile: float) -> float:
    return float(df[tag].dropna().quantile(percentile))


def percentage_missing(df: pd.DataFrame, tag: str) -> float:
    return float(df[tag].isna().mean() * 100)


def average_length_of_nan_chunks(df: pd.DataFrame, tag: str) -> float:
    is_nan = df[tag].isna()
    if not bool(is_nan.any()): # casting to bool cuz pandas is weird
        return 0.0

    nan_chunks = []
    current_chunk_length = 0

    for value in is_nan:
        if value:
            current_chunk_length += 1
        elif current_chunk_length > 0:
            nan_chunks.append(current_chunk_length)
            current_chunk_length = 0

    if current_chunk_length > 0:
        nan_chunks.append(current_chunk_length)

    if not nan_chunks:
        return 0.0

    return float(sum(nan_chunks) / len(nan_chunks))


def raw_data_eval_for_tag(df: pd.DataFrame, tag: str) -> dict:
    return {
        'mean': mean_by_tag(df, tag),
        'variance': variance_by_tag(df, tag),
        'max': max_by_tag(df, tag),
        'min': min_by_tag(df, tag),
        '50th_percentile': percentile_by_tag(df, tag, 0.50),
        '90th_percentile': percentile_by_tag(df, tag, 0.90),
        'percent_nan': percentage_missing(df, tag),
        'average_length_of_nan_chunks': average_length_of_nan_chunks(df, tag),
    }


@config()
class RawDataEvalConfig(BaseEvalConfig):
    name: str = 'raw_data'
    caller_codes: list[CallerCode] = field(default_factory=lambda:[CallerCode.ONLINE])
    stage_codes: list[StageCode] = field(default_factory=lambda:[StageCode.INIT])
    enabled: bool = True
    tags: list[str] = field(default_factory=list) # which tags you want to output stats for


def raw_data_eval(
        cfg: RawDataEvalConfig,
        app_state: AppState,
        pf: PipelineFrame,
    ) -> None:

    if not cfg.enabled:
        return

    df = pf.data
    for tag in cfg.tags:
        if tag not in df.columns:
            logging.warning(f"Tag {tag} not found in data frame columns.")
        else:
            stat_dict = raw_data_eval_for_tag(df, tag)
            for stat_name, stat_value in stat_dict.items():
                app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric=f'{tag}_{stat_name}',
                    value=stat_value,
                )
