
import logging
import shutil
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pydantic import Field
from tabulate import tabulate
from tqdm import tqdm

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import StageCode
from corerl.eval.plotting.report import (
    plot_chunk_histogram,
    plot_nan_histogram,
    plot_sensor_data,
    plot_sensor_histogram,
)
from corerl.eval.raw_data import raw_data_eval_for_tag

log = logging.getLogger(__name__)

@config()
class ReportConfig:
    output_dir: Path = Field(default_factory=lambda:'report/')
    stages : list[StageCode] = Field(default_factory=lambda:[StageCode.INIT])
    tags_to_exclude : list = Field(default_factory=list) # tags to exclude from analysis

    # for stat table
    stat_table_enabled : bool = True

    # for cross correlation
    # options for cross_corr_tags:
    # 1. list of tag names -> will find cross correlation for all pairs of tags in this list
    # 2. list of list[str] -> will find cross correlation only for these pairs in each list.
    # 3. None -> will find cross correlation for ALL pairs of tags
    cross_corr_enabled : bool = True
    cross_corr_tags : list[str] | list[list[str]] | None = Field(default_factory=list)
    cross_corr_max_lag : int = Field(default_factory=lambda: 100)

    # for histograms
    hist_enabled : bool = True
    hist_show_mean: bool = Field(default_factory=lambda: True)
    hist_percentiles: list[float] = Field(default_factory=lambda: [0.1, 0.9])
    hist_num_bins: int = Field(default_factory=lambda: 30)


def get_tags(data: list[pd.DataFrame]) -> list[str]:
    tags = list(data[0].columns)
    return tags


def make_stat_table(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        output_path: Path,
    ) -> None:

    if not cfg.stat_table_enabled:
        return

    tags = get_tags(data)
    table_data = []
    headers = ['tag']
    for tag in tags:
        if tag is cfg.tags_to_exclude:
            continue
        for stage_i, df in enumerate(data):
            row = [f"{tag} - {stages[stage_i].name}"]
            headers = ['tag']
            tag_stats = raw_data_eval_for_tag(df, tag)
            for key, value in tag_stats.items():
                headers.append(key)
                row.append(round(value, 2))
            table_data.append(row)
    table_data.insert(0, headers)
    table_str = tabulate(table_data, headers='firstrow', tablefmt='grid')
    with open(output_path/'sensor_report.txt', 'w') as f:
        f.write(table_str)


def make_distribution_plots(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        output_path: Path,
    ) -> None:

    if not cfg.hist_enabled:
        return

    tags = get_tags(data)

    show_hist_mean = cfg.hist_show_mean
    percentiles = cfg.hist_percentiles
    num_bins = cfg.hist_num_bins

    log.info('Generating Distribution Plots...')
    for tag in tqdm(tags):
        for stage_i, df in enumerate(data):
            stage_name = stages[stage_i].name
            tag_stage_output_path = output_path / tag / stage_name
            tag_stage_output_path.mkdir(parents=True, exist_ok=True)
            plot_sensor_data(
                df,
                tag,
                save_path= tag_stage_output_path / f'{tag}_sensor_data_{stage_name}.png',
                title=f'{tag} Sensor Data - {stage_name}',
            )
            plot_sensor_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_sensor_histogram_{stage_name}.png',
                title=f'{tag} Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles = percentiles,
                bins=num_bins,
            )
            plot_nan_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_nan_histogram_{stage_name}.png',
                title=f'{tag} NaN Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles = percentiles,
                bins=num_bins,
            )
            plot_chunk_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_chunk_histogram_{stage_name}.png',
                title=f'{tag} Chunk Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles = percentiles,
                bins=num_bins,
            )


def get_tag_pairs(
        cfg: ReportConfig,
        data: list[pd.DataFrame]
    ) -> list[tuple[str, str]]:
    tag_info = cfg.cross_corr_tags
    if tag_info is None: # all pairs of tags
        tags = get_tags(data)
        pairs = list(combinations(tags, 2))
    else:
        if len(tag_info) == 0:
            return []
        elif isinstance(tag_info[0], str): # all pairs of tags specified in cfg.cross_corr_tags
            str_tags = tag_info
            pairs = list(combinations(str_tags, 2))
        else: # only pairs of tags specified in cfg.cross_corr_tags
            for pair in tag_info:
                assert len(pair) == 2
            pairs = tag_info

    pairs = [tuple(pair) for pair in pairs]
    pairs = cast(list[tuple[str, str]], pairs)
    return pairs


def make_cross_correlation_table(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        output_path: Path,
        ) -> None:

    if not cfg.cross_corr_enabled:
        return

    log.info("Making cross correlation table")
    tag_pairs = get_tag_pairs(cfg, data)
    all_tags = get_tags(data)
    max_lag = cfg.cross_corr_max_lag
    table = [['Stage', 'tag 1', 'tag 2', 'Max Cross Correlation', 'Lag for Max Cross Correlation']]
    for stage_i, stage in enumerate(stages):
        df = data[stage_i]
        for tag_pair in tqdm(tag_pairs):
            tag_1, tag_2 = tag_pair

            assert tag_1 in all_tags
            assert tag_2 in all_tags
            cc, lag, _ = cross_correlation(df, tag_1, tag_2, max_lag)
            row = [stage.name, tag_1, tag_2, cc, lag]
            table.append(row) # type: ignore

    table_str = tabulate(table, headers='firstrow', tablefmt='grid')
    with open(output_path/'cross_correlation.txt', 'w') as f:
        f.write(table_str)


def standardize(x: np.ndarray, mask: np.ndarray):
    """
    Standardization that ignores all values not included in the mask.
    Masked values are output as 0.
    """
    if np.all(~mask):
        return np.zeros_like(x)
    std = np.std(x[mask])
    if std == 0:
        return np.zeros_like(x)
    x = x.copy()
    mean = np.mean(x[mask])
    return mask*((x-mean)/std)


def correlate(
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int,
    ) -> np.ndarray:
    """
    Unnormalized cross correlation for lags in [-max_lag, +max_lag]
    """
    assert (not np.any(np.isnan(x))) and (not np.any(np.isnan(y)))
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_, y_ = x[:lag], y[-lag:] #note: lag is negative, so we are implicity flipping the sign for indexing
        elif lag > 0:
            x_, y_ = x[lag:], y[:-lag]
        else:
            x_, y_ = x, y
        corr = np.correlate(x_, y_)
        assert len(corr) == 1
        corr = corr[0]
        correlations.append(corr)

    return np.array(correlations)


def cross_correlation(
        df: pd.DataFrame,
        tag_1: str,
        tag_2: str,
        max_lag: int
    ) -> tuple[float, float, np.ndarray]:
    """
    Computes cross correlation between tag_1 and tag_2, where the lag may vary between [-max_lag, +max_lag].

    Note that the returned variables can be greater than one.
    """

    x = df[tag_1].to_numpy()
    y = df[tag_2].to_numpy()

    x_mask = ~np.isnan(x) # where x is not nan
    y_mask = ~np.isnan(y) # where y is not nan

    if np.all(~x_mask) or np.all(~y_mask): # either x or y is all nan
        return -np.inf, 0, np.array([-np.inf])

    # zero-out nans
    x = np.where(np.isnan(x), 0, x)
    y = np.where(np.isnan(y), 0, y)

    # Standardize x and y, but only using statistics from non-nan entries.
    # Note that standardization is performed on the masked arrays,
    # not the subset of data that is passed to correlate.
    x, y = standardize(x, x_mask), standardize(y, y_mask)

    # Compute unnormalized cross correlation between x and y.
    # Since previous nans are now 0, they do not affect unnormalized result
    corr = correlate(x, y, max_lag)

    # for each lag, we can find the number of non-nan entries in the sum
    # by taking the cross correlation of x_mask and y_mask
    num_non_nan = correlate(x_mask.astype(int), y_mask.astype(int), max_lag)

    # make zeros nan, to signal when cross corr not valid
    num_non_nan = np.where(num_non_nan == 0, np.nan, num_non_nan)

    cross_corr = corr / num_non_nan # normalize cross corr
    cross_corr = np.where(np.isnan(cross_corr), -np.inf, cross_corr)
    max_idx = np.argmax(cross_corr)
    lags = np.arange(-max_lag, max_lag+1)
    return cross_corr[max_idx], lags[max_idx], cross_corr


def generate_report(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
    ) -> None:

    output_path = Path(cfg.output_dir)
    if output_path.exists():
        log.warning(f'Output path {output_path} already exists. Deleting...')
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    make_stat_table(cfg, data, stages, output_path)
    make_distribution_plots(cfg, data, stages, output_path/'plots')
    make_cross_correlation_table(cfg, data, stages, output_path)

