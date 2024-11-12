from typing import List, Tuple
import numpy as np
import pandas as pd
from pandas import Index

def fill_in_missing_data(
        data: pd.DataFrame,
        remove_mode: str ='Average',
        remove_length_threshold: float = np.inf # seconds
) -> pd.DataFrame:
    tags = data.columns.values.tolist()
    if remove_mode == 'Delete':
        data.dropna(axis=0, inplace=True)
    elif remove_mode == 'Zero':
        data.fillna(0., inplace=True)
    elif remove_mode == 'Copy':
        for tag in tags:
            column = data[tag]
            nan_idx: Index = pd.Index(column.index[column.apply(np.isnan)])
            for idx in nan_idx:
                if idx == 0:
                    data.loc[idx, tag] = 0.
                else:
                    data.loc[idx, tag] = data.loc[idx-1, tag].copy()
    elif remove_mode == 'Average':
        too_long_intervals = set()
        for tag in tags:
            column = data[tag]
            # assert  False, type(column.index[column.apply(np.isnan)])
            nan_idx: Index = pd.Index(column.index[column.apply(np.isnan)])
            replace_interval = _check_interval_length(nan_idx)
            for interval in replace_interval:
                s, e = interval
                if e - s > remove_length_threshold:
                    too_long_intervals.update(list(np.arange(s, e + 1)))
            data[tag].interpolate(method="linear", inplace=True)
        if len(too_long_intervals) > 0:
            for key in list(data.keys()):
                data[key] = data[key].astype(np.float32)
                data[key][too_long_intervals] = np.nan
            data = fill_in_missing_data(data, remove_mode='Delete')
    return data

def _check_interval_length(idx: Index) -> List[List]:
    replace_interval = []
    k = 0
    while k < len(idx):
        block_start = idx[k]
        block_end = -1
        found_end = False
        while not found_end and k < len(idx):
            if k + 1 == len(idx) or idx[k + 1] != idx[k] + 1:
                found_end = True
                block_end = idx[k]
            k += 1
        replace_interval.append([block_start, block_end])
    return replace_interval

def clean_by_multiple_sliding_window(
        data: np.ndarray,
        window_length_lst: List[int],
        in_range_threshold: float
) -> Tuple[np.ndarray, dict]:
    outlier_counts = np.zeros(len(data))
    infos = {}
    for window_length in window_length_lst:
        sw, info = _clean_by_sliding_window(
            data, window_length=window_length, in_range_threshold=in_range_threshold
        )
        outlier_counts[sw] += 1
        infos[window_length] = info
    outliers = np.where(outlier_counts >= 0.5*len(window_length_lst))[0]
    return outliers, infos

def _clean_by_sliding_window(
        data: np.ndarray,
        window_length: int,
        in_range_threshold: float,
        min_window_count: int = 12
) -> Tuple[np.ndarray, dict]:
    # if the window length is too long,
    # then jump and ensure one number is evaluated by at least 12 windows
    nans0 = np.empty(window_length - 1)
    nans1 = np.empty(window_length - 1)
    nans0[:] = np.nan
    nans1[:] = np.nan
    padded_data = np.concatenate([nans0, data, nans1], axis=0)
    skip = max(1, window_length // min_window_count)
    window_starts = np.arange(0, window_length, skip)
    window_idx = np.arange(0, window_length)
    data_idxs = np.arange(len(data)).reshape(-1, 1)
    data_idxs = data_idxs + window_starts
    data_idxs = data_idxs.reshape(data_idxs.shape[0], data_idxs.shape[1], 1)
    data_idxs = data_idxs + window_idx
    windows = padded_data[data_idxs]

    nan_mask = np.isnan(windows)
    masked_windows = np.ma.masked_array(windows, mask=nan_mask)
    window_mean = masked_windows.mean(axis=-1)
    window_std = masked_windows.std(axis=-1)

    upper_windows = window_mean + window_std * 1.96
    lower_windows = window_mean - window_std * 1.96

    in_range_count = np.zeros(len(data))
    for i in range(len(data)):
        d = data[i]
        uppers, lowers = upper_windows[i], lower_windows[i]
        in_ranges = np.where(np.logical_and(lowers <= d, d <= uppers))[0]
        in_range_count[i] = len(in_ranges)
    threshold = in_range_threshold * window_length
    outliers = np.where(in_range_count < threshold)[0]

    info = {
        "lower": lower_windows,
        "upper": upper_windows
    }
    return outliers, info
