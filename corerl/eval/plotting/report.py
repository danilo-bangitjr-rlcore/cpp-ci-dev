
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_sensor_data(
        df: pd.DataFrame,
        tag: str,
        save_path: Path,
        title: str = '',
        ):
    """
    Plots sensor data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[tag].dropna(), label=f'{tag} data')
    plt.title(f'{tag} Sensor Data')
    plt.xlabel('Index')
    plt.ylabel(tag)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_histogram(
        data: pd.Series,
        title: str,
        save_path: Path,
        xlabel: str,
        show_mean: bool = False,
        percentiles: Optional[List[float]] = None,
        bins: int = 30
    ) -> None:
    """
    Generates a histogram, with mean and percentiles shown.
    """
    percentiles = percentiles or []
    plt.figure(figsize=(12, 6))

    if data.empty:
        plt.text(
            0.5, 0.5,
            'No data',
            horizontalalignment='center',
            verticalalignment='center',
            transform=plt.gca().transAxes,
            fontsize=20,
            color='red',
        )
    else:
        plt.hist(data, bins=bins, alpha=0.7, label=f'{title} distribution')
    if show_mean:
        mean_value = data.mean()
        plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='r')
    for percentile in percentiles:
        perc_value = data.quantile(percentile)
        plt.axvline(perc_value, color='g', linestyle='dashed', linewidth=1)
        plt.text(perc_value, plt.ylim()[1] * 0.8, f'{percentile*100:.0f}th: {perc_value:.2f}', color='g')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig(save_path)
    plt.close()


def plot_sensor_histogram(
        df: pd.DataFrame,
        tag: str,
        save_path: Path,
        title: str = '',
        show_mean: bool = False,
        percentiles: Optional[List[float]] = None,
        bins: int = 30
    ) -> None:
    """
    Plots distribution of non-nan sensor values.
    """
    data = df[tag].dropna()
    plot_histogram(data, title, save_path, tag, show_mean, percentiles, bins)


def plot_nan_histogram(
        df: pd.DataFrame,
        tag: str,
        save_path: Path,
        title: str = '',
        show_mean: bool = False,
        percentiles: Optional[List[float]] = None,
        bins: int = 30
    ) -> None:
    """
    Plots histogram of length of contiguous chunks of nans.
    """
    is_nan = df[tag].isna()

    nan_chunks = is_nan.astype(int).groupby(is_nan.ne(is_nan.shift()).cumsum()).sum()
    nan_lengths = nan_chunks[nan_chunks > 0]
    plot_histogram(nan_lengths, title, save_path, 'NaN Chunk Length', show_mean, percentiles, bins)


def plot_chunk_histogram(
        df: pd.DataFrame,
        tag: str,
        save_path: Path,
        title: str = '',
        show_mean: bool = False,
        percentiles: Optional[List[float]] = None,
        bins: int = 30
    ) -> None:
    """
    Plots histogram of length of contiguous chunks of non-nans.
    """
    is_not_nan = df[tag].notna()

    chunks = is_not_nan.astype(int).groupby(is_not_nan.ne(is_not_nan.shift()).cumsum()).sum()
    lengths = chunks[chunks > 0]
    plot_histogram(lengths, title, save_path, 'Chunk Length', show_mean, percentiles, bins)

