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
    assert isinstance(is_nan, pd.Series)
    chunk_lengths = length_of_chunks(is_nan)
    return float(sum(chunk_lengths) / len(chunk_lengths))


def average_length_of_non_nan_chunks(df: pd.DataFrame, tag: str) -> float:
    is_non_nan = df[tag].notna()
    assert isinstance(is_non_nan, pd.Series)
    chunk_lengths = length_of_chunks(is_non_nan)
    return float(sum(chunk_lengths) / len(chunk_lengths))


def length_of_chunks(series: pd.Series) -> list[int]:
    """
    Returns list of chunk lengths, where chunks are defined by maximal sequences of True in series.
    """
    if not bool(series.any()): # casting to bool cuz pandas is weird
        return [0]

    chunk_lengths = []
    current_chunk_length = 0

    for value in series:
        if value:
            current_chunk_length += 1
        elif current_chunk_length > 0: # i.e. is not value==True and the current chunk length is greater than 0
            chunk_lengths.append(current_chunk_length)
            current_chunk_length = 0

    if current_chunk_length > 0:
        chunk_lengths.append(current_chunk_length)

    return chunk_lengths


def number_of_non_nan_samples(df: pd.DataFrame, tag: str) -> int:
    return int(df[tag].notna().sum())


def number_of_nan_samples(df: pd.DataFrame, tag: str) -> int:
    return int(df[tag].isna().sum())


def raw_data_eval_for_tag(df: pd.DataFrame, tag: str) -> dict:
    return {
        'num_non_nan' : number_of_non_nan_samples(df, tag),
        'num_nan' : number_of_nan_samples(df, tag),
        'mean': mean_by_tag(df, tag),
        'variance': variance_by_tag(df, tag),
        'max': max_by_tag(df, tag),
        'min': min_by_tag(df, tag),
        '50th_percentile': percentile_by_tag(df, tag, 0.50),
        '90th_percentile': percentile_by_tag(df, tag, 0.90),
        'percent_nan': percentage_missing(df, tag),
        'average_length_of_nan_chunks': average_length_of_nan_chunks(df, tag),
        'average_length_of_non_nan_chunks': average_length_of_non_nan_chunks(df, tag),
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
    ) -> dict[str, dict] | None:

    if not cfg.enabled:
        return

    result_dict = {}
    df = pf.data
    for tag in cfg.tags:
        if tag not in df.columns:
            logging.warning(f"Tag {tag} not found in data frame columns.")
        else:
            stat_dict = raw_data_eval_for_tag(df, tag)
            result_dict[tag] = stat_dict
            for stat_name, stat_value in stat_dict.items():
                app_state.metrics_writer.write(
                    agent_step=app_state.agent_step,
                    metric=f'{tag}_{stat_name}',
                    value=stat_value,
                )
    return result_dict
