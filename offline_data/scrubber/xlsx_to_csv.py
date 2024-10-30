import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pickle as pkl
import copy
import matplotlib.pyplot as plt

SENSORS = ["AIC-3730 (PID Setpoint)", "AIC-3730 (PID Process Value)", "AIC-3730 (PID Output)",
           "AIC-3731 (PID Setpoint)", "AIC-3731 (PID Process Value)", "AIC-3731 (PID Output)",
           "PDIC-3738 (PID Setpoint)", "PDIC-3738 (PID Process Value)", "PDIC-3738 (PID Output)", "PI-0169",
           "LI-3734 ", "FI-0887", "FI-0872",
           "AI-0879A", "AI-0879B",
           "TI-0880", "AI-0897B"]
FORMAL_SENSOR_NAMES = {
    "AIC-3730 (PID Setpoint)": "pH SP",
    "AIC-3730 (PID Process Value)": "pH PV",
    "AIC-3730 (PID Output)": "pH Output",
    "AIC-3731 (PID Setpoint)": "ORP SP",
    "AIC-3731 (PID Process Value)": "ORP PV",
    "AIC-3731 (PID Output)": "ORP Output",
    "PDIC-3738 (PID Setpoint)": "Differential Pressure SP",
    "PDIC-3738 (PID Process Value)": "Differential Pressure PV",
    "PDIC-3738 (PID Output)": "Differential Pressure Output",
    "PI-0169": "Blower Discharger Pressure",
    "LI-3734 ": "Scrubber Tower Sump Level",
    "FI-0887": "Scrubber Softened Water Flow",
    "FI-0872": "Scrubber Recirculation Flow",
    "AI-0879A": "Inlet H2S",
    "AI-0879B": "Outlet H2S",
    "Efficiency": "Efficiency",
    "TI-0880": "Inlet Temperature",
    "AI-0897B": "Softened Water Conductivity",
    "FV-3735": " Mist Eliminator Valve Status",
    "M-3739": "Blower Status",
}
EQUALNAMES = {
    "FIC-3734 (PID Process Value)": "FI-0887",
    "FIC-3734 (PID Setpoint)": "FI-0887 (Estimated Setpoint)",
}


def fill_in_missing_data(data, str_column='Timestamp', remove_mode='Average', remove_length_threshold=np.inf):
    keys = list(data.keys())

    if remove_mode == 'Delete':
        remove_idx = set()
        for key in keys:
            if key != str_column:
                nan_idx = np.where(np.isnan(data[key]))[0]
                for i in nan_idx:
                    remove_idx.add(i)
            else:
                for i, s in enumerate(data[key]):
                    if type(s) == np.float64 and np.isnan(s):
                        remove_idx.add(i)

        remove_idx = list(remove_idx)
        remove_idx.sort()
        keep_idx = list(np.arange(len(list(data.values())[0])))
        end_of_ep = np.zeros(len(keep_idx))

        for i in remove_idx:
            keep_idx.remove(i)
            end_of_ep[i-1] = 1
            end_of_ep[i] = 1
        # end_of_ep[-1] = 1
        for key in keys:
            data[key] = data[key][keep_idx]
        end_of_ep = end_of_ep[keep_idx].astype(int)

    elif remove_mode == 'Zero':
        end_of_ep = np.zeros(len(data[list(data.keys())[0]]))
        remove_idx = set()
        for key in keys:
            if key != str_column:
                nan_idx = np.where(np.isnan(data[key]))[0]
                for i in nan_idx:
                    remove_idx.add(i)
            else:
                for i, s in enumerate(data[key]):
                    if type(s) == np.float64 and np.isnan(s):
                        remove_idx.add(i)
        for i in remove_idx:
            for key in keys:
                data[key][i] = 0
        end_of_ep = end_of_ep.astype(int)

    elif remove_mode == 'Copy':
        remove_idx = set()
        for key in keys:
            if key != str_column:
                nan_idx = np.where(np.isnan(data[key]))[0]
                for i in nan_idx:
                    remove_idx.add(i)
            else:
                for i, s in enumerate(data[key]):
                    if type(s) == np.float64 and np.isnan(s):
                        remove_idx.add(i)
        for i in remove_idx:
            for key in keys:
                data[key][i] = data[key][i-1]
        end_of_ep = end_of_ep.astype(int)

    elif remove_mode == 'Average':
        too_long_intervals = set()
        for key in keys:
            remove_idx = set()
            if key != str_column:
                nan_idx = np.where(np.isnan(data[key]))[0]
                for i in nan_idx:
                    remove_idx.add(i)
            remove_idx = list(remove_idx)
            remove_idx.sort()
            data[key], end_of_ep, replace_interval = array_replace_by_average(data[key], remove_idx)
            for interval in replace_interval:
                s, e = interval
                if e - s > remove_length_threshold:
                    too_long_intervals.update(list(np.arange(s, e+1)))

        end_of_ep = np.zeros(data[keys[0]].shape[0]).astype(int)
        too_long_intervals = list(too_long_intervals)

        if len(too_long_intervals) > 0:
            for key in list(data.keys()):
                if key in ["Timestamp"]:
                    data[key][too_long_intervals] = pd.NA
                else:
                    data[key] = data[key].astype(np.float32)
                    data[key][too_long_intervals] = np.nan
            data, end_of_ep = fill_in_missing_data(data, str_column=str_column, remove_mode='Delete')

    return data, end_of_ep

def shutdown_to_nan(columns, sensor, threshold):
    data = columns[sensor]
    to_nan = np.where(data < threshold)[0]
    to_nan_delay = to_nan + 4
    to_nan = np.unique(np.concatenate((to_nan, to_nan_delay), 0))
    # columns[sensor][to_nan] = np.nan
    columns.loc[to_nan, sensor] = np.nan
    return columns, to_nan

def date_range_to_nan(columns, change_list, time_start, time_end):
    timestamp = columns['Timestamp']
    to_nan = np.where(np.logical_and(timestamp>=time_start, timestamp<=time_end))[0]
    for sensor in change_list:
        # columns[sensor][to_nan] = np.nan
        columns.loc[to_nan, sensor] = np.nan
    return columns, to_nan

def remove_date_range(columns, time_start, time_end):
    timestamp = columns['Timestamp']
    # remove = np.where(np.logical_and(timestamp>=time_start, timestamp<=time_end))[0]
    remove = np.where(np.logical_and(timestamp>=time_start, timestamp<=time_end))[0]
    # print(remove, len(columns))
    # print(columns.iloc[remove]['Timestamp'])
    columns = columns.drop(remove)
    # print(len(columns))
    columns = columns.reset_index(drop=True)
    # print(columns.iloc[remove]['Timestamp'])
    return columns

def clean_by_sliding_window(data, window_length, in_range_threshold, min_window_count=12):
    skip = max(1, window_length // min_window_count) # if the window length is too long, then jump and ensure one number is evaluated by at least 12 windows
    nans0 = np.empty(window_length - 1)
    nans1 = np.empty(window_length - 1)
    nans0[:] = np.nan
    nans1[:] = np.nan
    temp_data = np.concatenate([nans0, data, nans1], axis=0)
    window_starts = np.arange(0, window_length, skip)
    window_idx = np.arange(0, window_length)
    data_idxs = np.arange(len(data)).reshape(-1, 1)
    data_idxs = data_idxs + window_starts
    data_idxs = data_idxs.reshape(data_idxs.shape[0],data_idxs.shape[1],1)
    data_idxs = data_idxs + window_idx
    windows = temp_data[data_idxs]

    nan_mask = np.isnan(windows)
    masked_windows = np.ma.masked_array(windows, mask=nan_mask)
    real_num_count = window_length - nan_mask.astype(int).sum(axis=-1)
    temp_mean = masked_windows.mean(axis=-1)
    temp_ste = masked_windows.std(axis=-1) #/ np.sqrt(real_num_count)

    upper_windows = temp_mean + temp_ste * 1.96
    lower_windows = temp_mean - temp_ste * 1.96

    in_range_count = np.zeros(len(data))
    for i in range(len(data)):
        d = data[i]
        uppers, lowers = upper_windows[i], lower_windows[i]
        in_ranges = np.where(np.logical_and(lowers<=d, d<=uppers))[0]
        in_range_count[i] = len(in_ranges)
    threshold = in_range_threshold * window_length
    outliers = np.where(in_range_count < threshold)[0]

    info = {
        "lower": lower_windows,
        "upper": upper_windows
    }
    return outliers, info

def clean_by_multiple_sliding_window(data, window_length_lst, in_range_threshold):
    outlier_counts = np.zeros(len(data))
    infos = {}
    for window_length in window_length_lst:
        sw, info = clean_by_sliding_window(data, window_length=window_length, in_range_threshold=in_range_threshold)
        outlier_counts[sw] += 1
        infos[window_length] = info
    outliers = np.where(outlier_counts >= 0.5*len(window_length_lst))[0]
    return outliers, infos

def save_window_conditions(sensor_wrong, columns, window_length_hardcode, in_range_threshold, filename):
    sliding_window_info = {}
    for sensor in columns:
        if sensor not in window_length_hardcode:
            continue
        print("saving window cleaning for column", sensor)
        sw, info = clean_by_multiple_sliding_window(columns[sensor], window_length_lst=window_length_hardcode[sensor], in_range_threshold=in_range_threshold)
        sensor_wrong[sensor].update(sw)
        sliding_window_info[sensor] = {
            "info": info,
            "outlier_idx": sw
        }
    with open("data_cleaning/scrubber{}.pkl".format(filename), "wb") as f:
        pkl.dump(sliding_window_info, f)
    return sensor_wrong


def array_replace_by_average(ary_in, remove_idx):
    ary = copy.deepcopy(ary_in)
    keep_idx = list(np.arange(len(ary)))
    end_of_ep = np.zeros(len(keep_idx))
    replace_interval = []
    k = 0
    while k < len(remove_idx):
        block_start = remove_idx[k]
        found_end = False
        while not found_end and k < len(remove_idx):
            if k + 1 == len(remove_idx) or remove_idx[k + 1] != remove_idx[k] + 1:
                found_end = True
                block_end = remove_idx[k]
            k += 1
        replace_interval.append([block_start, block_end])
        if block_end == len(end_of_ep) - 1:
            for i in range(block_start, block_end + 1):
                ary[i] = ary[i - 1]
        else:
            after = ary[block_end + 1]
            if block_start == 0:
                before = after
            else:
                before = ary[block_start - 1]
            step = (after - before) / (block_end - block_start + 1)
            assert not np.isnan(step), print(after, before, block_end, block_start)
            for t in range(block_start, block_end + 1):
                ary[t] = before + step * (t - block_start + 1)
    return ary, end_of_ep, replace_interval

def load_window_conditions(sensor_wrong, columns, filename):
    with open("data_cleaning/scrubber{}.pkl".format(filename), "rb") as f:
        sliding_window_info = pkl.load(f)
    for sensor in columns:
        if sensor not in sliding_window_info:
            continue
        sw = sliding_window_info[sensor]["outlier_idx"]
        sensor_wrong[sensor].update(sw)

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        xs = np.arange(len(columns[sensor]))
        replaced_data, _, _ = array_replace_by_average(columns[sensor], sw)
        cutoffs = [[0, len(xs)],
                   [329151, len(xs)]]
        for id_, co in enumerate(cutoffs):
            axs[id_].plot(xs[co[0]: co[1]], columns[sensor][co[0]: co[1]], zorder=5, label="Raw Data")
            # axs[id_].fill_between(xs[co[0]: co[1]], lower_windows.mean(axis=1)[co[0]: co[1]], upper_windows.mean(axis=1)[co[0]: co[1]], alpha=0.3,
            #                     linewidth=0., color='grey', zorder=1)
            axs[id_].plot(xs[co[0]: co[1]], replaced_data[co[0]: co[1]], zorder=5, label="Average replacement")

            clip_x = list(range(co[0], co[1], (co[1]-co[0])//10))
            axs[id_].set_xticks(clip_x)
            axs[id_].set_xticklabels(clip_x)

        axs[0].title.set_text(FORMAL_SENSOR_NAMES[sensor])

        plt.legend()
        plt.tight_layout()
        plt.show()
    return sensor_wrong

def wrong_efficiency(df):
    status_wrong = set() # all sensors are considered wrong
    # condition: inlet and output are positive
    eff_wrong = np.where(df["AI-0879A"]<=0)[0]
    status_wrong.update(list(eff_wrong))
    eff_wrong = np.where(df["AI-0879B"]<0)[0]
    status_wrong.update(list(eff_wrong))

    # condition: inlet not less than output
    eff_wrong = np.where(df["AI-0879A"] < df["AI-0879B"])[0]
    status_wrong.update(list(eff_wrong))
    return status_wrong

def scrubber_data_clean(df, window_length_hardcode, save=True, clean_data_file=''):
    # condition: efficiency is out of range [0, 1]
    # do not replace these data, use it for trace calculation, but not use them for training
    trace_cal_only = set()
    we = wrong_efficiency(df)
    trace_cal_only.update(list(we))

    status_wrong = set() # all sensors are considered wrong
    sensor_wrong = dict() # a single sensor is considered wrong
    for sensor in df:
        sensor_wrong[sensor] = set()

    # condition: remove data when fan is shutdown
    df, sys_off = shutdown_to_nan(df, 'PDIC-3738 (PID Process Value)', 1)
    status_wrong.update(list(sys_off))

    # condition: remove data at specific date
    df, sys_off = date_range_to_nan(df, ['PDIC-3738 (PID Process Value)'],
                                   datetime.datetime.strptime('6/17/2024 9:16:00', "%m/%d/%Y %H:%M:%S"),
                                   datetime.datetime.strptime('6/20/2024 15:31:00', "%m/%d/%Y %H:%M:%S"))
    status_wrong.update(list(sys_off))

    df, sys_off = date_range_to_nan(df, ['PDIC-3738 (PID Process Value)'],
                                   datetime.datetime.strptime('7/10/2024 19:01:00', "%m/%d/%Y %H:%M:%S"),
                                   datetime.datetime.strptime('7/17/2024 19:01:00', "%m/%d/%Y %H:%M:%S"))
    status_wrong.update(list(sys_off))

    if save:
        sensor_wrong = save_window_conditions(sensor_wrong, df, window_length_hardcode,
                                              in_range_threshold=0.1, filename=clean_data_file)
    else:
        sensor_wrong = load_window_conditions(sensor_wrong, df, clean_data_file)

    df = replace_bad_sensor(df, status_wrong, sensor_wrong, trace_cal_only)

    # Remove bad data.
    # It has to stay after the replace_bad_sensor function, to maintain a consistent row index.
    df = remove_date_range(df, datetime.datetime.strptime('5/21/2024 9:59:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('5/21/2024 10:00:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('5/21/2024 10:04:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('5/21/2024 10:05:30', "%m/%d/%Y %H:%M:%S"))

    df = remove_date_range(df, datetime.datetime.strptime('7/24/2024 7:45:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('7/24/2024 8:16:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('8/06/2024 23:25:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('8/06/2024 23:51:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('8/08/2024 08:15:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('8/08/2024 08:26:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('8/13/2024 23:50:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('8/13/2024 23:59:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('8/20/2024 08:15:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('8/20/2024 08:26:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('8/22/2024 03:00:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('8/22/2024 03:21:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('8/22/2024 14:55:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('8/22/2024 15:01:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('9/4/2024 13:05:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('9/4/2024 13:21:30', "%m/%d/%Y %H:%M:%S")) # mode=8
    df = remove_date_range(df, datetime.datetime.strptime('9/12/2024 15:35:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('9/12/2024 15:40:30', "%m/%d/%Y %H:%M:%S")) # mode=Null
    df = remove_date_range(df, datetime.datetime.strptime('9/12/2024 16:40:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('9/12/2024 16:45:30', "%m/%d/%Y %H:%M:%S")) # mode=Null
    df = remove_date_range(df, datetime.datetime.strptime('9/16/2024 8:10:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('9/16/2024 14:00:30', "%m/%d/%Y %H:%M:%S")) # H2S_in turned off
    df = remove_date_range(df, datetime.datetime.strptime('9/19/2024 7:30:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('9/16/2024 9:21:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/02/2024 9:55:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/02/2024 10:11:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/05/2024 2:45:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/05/2024 4:01:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/05/2024 18:55:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/05/2024 18:56:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/07/2024 07:20:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/07/2024 07:36:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/07/2024 09:45:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/07/2024 09:56:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/07/2024 09:45:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/07/2024 10:46:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/08/2024 01:15:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/08/2024 02:36:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/15/2024 09:50:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/15/2024 02:51:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/15/2024 11:05:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/15/2024 11:21:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/15/2024 13:25:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/15/2024 13:26:30', "%m/%d/%Y %H:%M:%S"))
    df = remove_date_range(df, datetime.datetime.strptime('10/20/2024 21:20:30', "%m/%d/%Y %H:%M:%S"),
                              datetime.datetime.strptime('10/20/2024 22:56:30', "%m/%d/%Y %H:%M:%S"))
    return df

def replace_bad_sensor(df, status_wrong, sensor_wrong, trace_cal_only):
    # replace bad sensor readings with NaN
    status_wrong = list(status_wrong)
    status_wrong.sort()
    for label in df:
        if label != 'Timestamp':
            df.loc[status_wrong , label] = np.nan
            if label in sensor_wrong:
                df.loc[list(sensor_wrong[label]) , label] = np.nan
    trace_cal_only = list(trace_cal_only)
    trace_cal_only.sort()
    df["BadDataIndicator"] = np.zeros(len(df.index))
    df.loc[trace_cal_only, "BadDataIndicator"] = 1
    return df

def raw2array(file_name, sensors,
              ignore_sensors=['ARC-3730 (PID External Setpoint)',
                              'ARC-3731 (PID External Setpoint)',
                              'PDRC-3738 (PID External Setpoint)',
                              'FRC-3734 (External Setpoint)']):
    xl_file = pd.ExcelFile(file_name)
    dfs = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}

    # remove external idx
    for sheet in dfs:
        drop_cols = []
        columns = dfs[sheet].iloc[1]
        for i, col in enumerate(columns):
            if col in ignore_sensors:
                drop_cols += list(range(i, i + 3))
        if len(drop_cols) > 0:
            dfs[sheet] = dfs[sheet].drop(dfs[sheet].columns[drop_cols], axis=1)
            new_len = len(dfs[sheet].columns)
            dfs[sheet].columns = ['Unnamed: {}'.format(int(i)) for i in range(new_len)]

    columns = {}
    status = {}
    timestamp_base = list(dfs[list(dfs.keys())[0]]["Unnamed: {}".format(0)])[3:]

    for k in dfs.keys():
        for ki in range(len(dfs[k].keys())):
            k_name = "Unnamed: {}".format(ki)
            dfs_column = list(dfs[k][k_name])
            if dfs_column[1] in sensors or dfs_column[1] in EQUALNAMES.keys():
                col_name = EQUALNAMES.get(dfs_column[1], dfs_column[1])

                timestamp = list(dfs[k]["Unnamed: {}".format(ki)])[3:]
                for t in range(len(timestamp)):
                    assert np.abs(timestamp[t] - timestamp_base[t]) < timedelta(minutes=2), print(k, t, timestamp[t], timestamp_base[t])
                value_column = list(dfs[k]["Unnamed: {}".format(ki+1)])
                columns[col_name] = np.array(value_column[3:])
                status_column = list(dfs[k]["Unnamed: {}".format(ki+2)])
                status[col_name] = status_column[3:]

                # check label, replace bad data by NaN
                for i, l in enumerate(status[col_name]):
                    if type(l) != str or "Good" not in l:
                        columns[col_name][i] = np.nan
    ts = dfs[list(dfs.keys())[0]]["Unnamed: {}".format(0)][3:]
    columns["Timestamp"] = np.array(ts)

    # sometimes the length of "TI-0880", "AI-0897B" are shorter. cut the length of other sensors
    check_range = []
    for k in columns:
        check_range.append(len(columns[k]))
    min_length = np.array(check_range).min()
    for k in columns:
        columns[k] = columns[k][:min_length]

    df = pd.DataFrame.from_dict(columns)
    # df = df.set_index("Timestamp")
    return df

def h2s_efficiency(data):
    reasonable = np.where(np.logical_and(
        np.logical_and(data["AI-0879A"]>0, data["AI-0879B"]>=0), data["AI-0879A"] > data["AI-0879B"]))[0]

    eff = 1 - data["AI-0879B"][reasonable] / data["AI-0879A"][reasonable]
    print("Average efficiency", eff.mask(eff < 0).mean())
    ideal_out = np.ones(len(reasonable))
    ideal_eff = 1 - ideal_out / data["AI-0879A"][reasonable]
    print("One ppm efficiency", ideal_eff.mask(ideal_eff < 0).mean())

    eff_all = 1 - data["AI-0879B"] / data["AI-0879A"]
    unreasonable = np.where(eff_all < 0)[0]
    eff_all[unreasonable] = 0#np.nan
    unreasonable = np.where(eff_all > 1)[0]
    eff_all[unreasonable] = 0#np.nan

    eff_all *= 100.
    return eff_all

def estimate_setpoint(data, setpoints):
    num_sp = len(setpoints)
    setpoints = np.array(setpoints).reshape(1, -1)
    setpoints = np.repeat(setpoints, len(data), axis=0)
    data = data.reshape(-1, 1)
    data = np.repeat(data, num_sp, axis=1)
    abs_diff = np.abs(data - setpoints)
    closest = np.argmin(abs_diff, axis=1)
    estimation = setpoints[np.arange(len(closest)), closest]
    return estimation.astype(np.float32)


def transfer(fxs_names, outlier_window_length):
    dfs = []
    for f in fxs_names:
        print("Transferring {}.xlsx".format(f))
        df = raw2array("{}.xlsx".format(f), SENSORS)
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)

    # Add efficiency
    dfs["AI-0879B"] = dfs["AI-0879B"] / 1000.
    dfs = scrubber_data_clean(dfs, outlier_window_length, save=True, clean_data_file="_new")
    dfs['Efficiency'] = h2s_efficiency(dfs)
    dfs["AI-0879B"] = dfs["AI-0879B"] * 1000. # get back to normal range

    # Add estimated flow rate setpoint
    if 'FIC-3734 (PID Setpoint)' not in dfs.columns:
        dfs['FI-0887 (Estimated Setpoint)'] = estimate_setpoint(dfs['FI-0887'].to_numpy(),
                                                                [15, 20, 25, 30, 35, 40, 45, 50])
    else:
        dfs['FI-0887 (Estimated Setpoint)'] = dfs['FIC-3734 (PID Setpoint)']
        dfs['FI-0887'] = dfs['FIC-3734 (PID Process Value)']
    return dfs

if __name__ == "__main__":

    window_length_hardcode = {
        "AIC-3730 (PID Setpoint)": [36],
        "AIC-3730 (PID Process Value)": [72],
        "AIC-3730 (PID Output)": [36],
        "AIC-3731 (PID Setpoint)": [12],
        "AIC-3731 (PID Process Value)": [72],
        "AIC-3731 (PID Output)": [36],
        "PDIC-3738 (PID Setpoint)": [24],
        "PDIC-3738 (PID Process Value)": [36],
        "PDIC-3738 (PID Output)": [36],
        "PI-0169": [72],
        "LI-3734 ": [12],
        "FI-0887": [3],
        "FI-0872": [6],
        "AI-0879A": [1],
        "AI-0879B": [1],
        "TI-0880": [12],
        "AI-0897B": [12]
    }

    # fxs = os.listdir('xlsx/')
    fxs = [
        'Scrubber 4 DV Data Export Data Only 2021.xlsx',
        'Scrubber 4 DV Data Export Data Only 2022.xlsx',
        'Scrubber 4 DV Data Export Data Only Jan-Jun 2023.xlsx',
        'Scrubber 4 DV Data Export Data Only July 2023-2024.xlsx',
        'Scrubber 4 DV Data Export Data Only May-6_13 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only May-13_21 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only May-21_27 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only May-27_Jun-03 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jun-04_Jun-09 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jun-10_Jun-16 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jun-17_Jun-23 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jun-24_Jul-01 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jul-02_Jul-09 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jul-10_Jul-14 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jul-15_Jul-22 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jul-23_Jul-28 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Jul-29_Aug-06 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Aug-07_Aug-13 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Aug-14_Aug-18 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Aug-19_Aug-26 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Aug-27_Sep-02 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Sep-03_Sep-08 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Sep-09_Sep-17 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Sep-18_Sep-22 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Sep-23_Sep-30 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Oct-01_Oct-04 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Oct-05_Oct-15 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Oct-16_Oct-20 2024.xlsx',
        'Scrubber 4 DV Data Export Data Only Oct-21_Oct-27 2024.xlsx',
    ]
    fxs_names = [os.path.join('xlsx', f.strip(".xlsx")) for f in fxs]
    df = transfer(fxs_names, window_length_hardcode)

    data_dict = df.to_dict(orient='list')
    for k in data_dict:
        data_dict[k] = np.array(data_dict[k])
    data_dict, _ = fill_in_missing_data(data_dict, 'Timestamp', 'Average', remove_length_threshold=12 * 24)

    # double check efficiency, label the data if the efficiency still doesn't meet request
    df = pd.DataFrame(data_dict)
    trace_cal_only = set()
    we = wrong_efficiency(df)
    trace_cal_only.update(list(we))
    df = replace_bad_sensor(df, set(), set(), trace_cal_only)
    data_dict = df.to_dict(orient='list')
    for k in data_dict:
        data_dict[k] = np.array(data_dict[k])
    data_dict, _ = fill_in_missing_data(data_dict, 'Timestamp', 'Delete', remove_length_threshold=12 * 24)

    df = pd.DataFrame.from_dict(data_dict)
    print("Column heads:")
    print("[\""+ "\", \"".join(df.columns.values) + "\"]")
    df.to_csv("clean_data.csv", index=False)
    # df.to_csv("clean_data_test.csv", index=False)

    new_fcs = os.listdir('.')
    print("[\"" + "\", \"".join(new_fcs) + "\"]")
