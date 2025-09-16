from pathlib import Path

from corerl.data_pipeline.datatypes import StageCode
import jax
import matplotlib.pyplot as plt
import pandas as pd
from corerl.eval.raw_data import length_of_chunks

from coreoffline.config import ReportConfig
from coreoffline.data_analysis.data_report import get_tags, log


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
        percentiles: list[float] | None = None,
        bins: int = 30,
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
        mean_value = float(data.mean())
        plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='r')
    for percentile in percentiles:
        perc_value = data.quantile(percentile)
        plt.axvline(perc_value, color='g', linestyle='dashed', linewidth=1)
        plt.text(perc_value, plt.ylim()[1] * 0.8, f'{percentile * 100:.0f}th: {perc_value:.2f}', color='g')

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
        percentiles: list[float] | None = None,
        bins: int = 30,
    ) -> None:
    """
    Plots distribution of non-nan sensor values.
    """
    data = df[tag].dropna()
    assert isinstance(data, pd.Series)
    plot_histogram(data, title, save_path, tag, show_mean, percentiles, bins)


def plot_nan_histogram(
        df: pd.DataFrame,
        tag: str,
        save_path: Path,
        title: str = '',
        show_mean: bool = False,
        percentiles: list[float] | None = None,
        bins: int = 30,
    ) -> None:
    """
    Plots histogram of length of contiguous chunks of nans.
    """
    is_nan = df[tag].isna()
    assert isinstance(is_nan, pd.Series)
    chunk_lengths = length_of_chunks(is_nan)
    chunk_lengths = pd.Series(chunk_lengths)
    plot_histogram(chunk_lengths, title, save_path, 'NaN Chunk Length', show_mean, percentiles, bins)


def plot_chunk_histogram(
        df: pd.DataFrame,
        tag: str,
        save_path: Path,
        title: str = '',
        show_mean: bool = False,
        percentiles: list[float] | None = None,
        bins: int = 30,
    ) -> None:
    """
    Plots histogram of length of contiguous chunks of non-nans.
    """
    is_not_nan = df[tag].notna()
    assert isinstance(is_not_nan, pd.Series)
    chunk_lengths = length_of_chunks(is_not_nan)
    chunk_lengths = pd.Series(chunk_lengths)
    plot_histogram(chunk_lengths, title, save_path, 'Chunk Length', show_mean, percentiles, bins)


def make_actor_critic_plots(
    eval_date: str,
    x_axis_actions: jax.Array,
    probs: jax.Array,
    qs: jax.Array,
    step: int,
    eval_state_num: int,
    save_path: Path,
):
    probs_a_dims, _ = probs.shape
    qs_a_dims, _ = qs.shape
    assert probs_a_dims == qs_a_dims

    fig, axs = plt.subplots(2, qs_a_dims, sharex=True, figsize=(5 * qs_a_dims, 10))
    for a_dim in range(qs_a_dims):
        axs[0, a_dim].set_title("Actor", fontsize=18)
        axs[0, a_dim].plot(x_axis_actions, probs[a_dim])
        axs[1, a_dim].set_title("Critic", fontsize=18)
        axs[1, a_dim].plot(x_axis_actions, qs[a_dim])

    fig.suptitle(eval_date, fontsize=32)
    fig.tight_layout()

    fig.savefig(save_path / f"offline_iter_{step}_state_{eval_state_num}_actor_critic_plot.png")
    plt.close(fig)


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
    for tag in tags:
        for stage_i, df in enumerate(data):
            stage_name = stages[stage_i].name
            tag_stage_output_path = output_path / tag / stage_name
            tag_stage_output_path.mkdir(parents=True, exist_ok=True)
            plot_sensor_data(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_sensor_data_{stage_name}.png',
                title=f'{tag} Sensor Data - {stage_name}',
            )
            plot_sensor_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_sensor_histogram_{stage_name}.png',
                title=f'{tag} Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles=percentiles,
                bins=num_bins,
            )
            plot_nan_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_nan_histogram_{stage_name}.png',
                title=f'{tag} NaN Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles=percentiles,
                bins=num_bins,
            )
            plot_chunk_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_chunk_histogram_{stage_name}.png',
                title=f'{tag} Chunk Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles=percentiles,
                bins=num_bins,
            )
