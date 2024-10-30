import pandas as pd
import numpy as np


def find_manual_mode(data_list):
    for file_name in data_list:
        file_name = "xlsx/" + file_name
        xl_file = pd.ExcelFile(file_name)
        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}
        timestamp = dfs[list(dfs.keys())[0]]["Unnamed: {}".format(0)][2:]
        ph_modes = dfs[list(dfs.keys())[0]]["Unnamed: {}".format(1)][2:]
        orp_modes = dfs[list(dfs.keys())[0]]["Unnamed: {}".format(2)][2:]
        # manual_idx = np.where(np.logical_or(ph_modes==8, orp_modes==8))[0]
        # if len(manual_idx) > 0 :
        #     print("In manual mode:\n{}".format(timestamp[manual_idx]))
        unknown_idx = np.where(np.logical_and(
            np.logical_or(ph_modes!=16, orp_modes!=16),
            np.logical_or(ph_modes!=32, orp_modes!=32)
        ))[0]
        if len(unknown_idx) > 0 :
            print("In unknown mode:\n{}".format(timestamp[unknown_idx]))


if __name__ == "__main__":
    mode_data = [
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Jul-10_Jul-14 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Jul-23_Jul-28 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Jul-29_Aug-06 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Aug-07_Aug-13 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Aug-14_Aug-18 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Aug-19_Aug-26 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Aug-27_Sep-02 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Sep-03_Sep-08 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Sep-09_Sep-17 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Sep-18_Sep-22 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Sep-23_Sep-30 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Oct-01_Oct-04 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Oct-05_Oct-15 2024.xlsx',
        # 'Scrubber 4 DV Data Export Controllers Mode Data Only Oct-16_Oct-20 2024.xlsx',
        'Scrubber 4 DV Data Export Controllers Mode Data Only Oct-21_Oct-27 2024.xlsx',
    ]
    find_manual_mode(mode_data)
