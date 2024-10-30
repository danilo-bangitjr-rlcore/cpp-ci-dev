import numpy as np
import pandas as pd


dst_start_time = {
    2020: pd.Timestamp("2021-03-08 02:00:00"),
    2021: pd.Timestamp("2021-03-14 02:00:00"),
    2022: pd.Timestamp("2022-03-13 02:00:00"),
    2023: pd.Timestamp("2023-03-12 02:00:00"),
    2024: pd.Timestamp("2024-03-10 02:00:00"),
}

dst_end_time = {
    2020: pd.Timestamp("2021-11-01 01:00:00"),
    2021: pd.Timestamp("2021-11-07 01:00:00"),
    2022: pd.Timestamp("2022-11-06 01:00:00"),
    2023: pd.Timestamp("2023-11-05 01:00:00"),
    2024: pd.Timestamp("2024-11-03 01:00:00"),
}


def in_dst(time):
    start = dst_start_time[time.year]
    end = dst_end_time[time.year]

    return start <= time < end


df = pd.read_csv("clean_data.csv")
times = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")

# For debugging
# times = times[71260:71300]    # DST Ends
# times = times[2735:2750]      # DST Begins

dt = 5  # Minutes

start_interval = times[0:60 // dt]
if start_interval.iloc[0] == start_interval.iloc[-1]:
    dst = True
else:
    dst = in_dst(start_interval.iloc[0])

forward_search = True
adjusted_times = []
for i, time in enumerate(times):
    if in_dst(time):
        dst = True
    elif dst_end_time[time.year] < time:  # After DST
        if dst and (
            dst_end_time[time.year].month == time.month and
            dst_end_time[time.year].day == time.day
        ):  # On the day of DST
            # If the rows of the df that should account for the next 1.5 hours
            # never go back in time, then we are no longer in DST. Otherwise,
            # we are at the first hour in the repeated hour of DST.
            ti = np.array(times[i:(i + 90//dt)])
            if time == times.iloc[i + 60 // dt] or (ti < time).any():
                dst = True
            else:
                dst = False

    adjusted_times.append(time.tz_localize("-0700" if not dst else "-0600"))


utc_times = np.array(
    list(
        map(
            lambda x: x.tz_convert("UTC").tz_convert(None),
            adjusted_times,
        ),
    ),
)
df["Timestamp"] = utc_times
df.set_index("Timestamp")
print(df["Timestamp"])
df.to_csv("clean_data_utc.csv", index=False, index_label="Timestamp")
