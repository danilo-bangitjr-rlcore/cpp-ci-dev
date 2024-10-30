# add new controllers mode data file name to the list in find_manual_model.py then run the following command. You may comment out the old ones.
# find_manual_mode.py will output all timestamps that mode doesn't equal to 16
python find_manual_mode.py

# please fill in the given timestamp ranges into scrubber_data_clean() function in xlsx_to_csv.py
# and fill in the new data file name to the fxs list in xlsx_to_csv.py
# then run the following command
python xlsx_to_csv.py

# the following cmd corrects the timestamp to use utc consistently
python adjust_dates.py

# the following cmd corrects the column name in csv
cd ../..
python rename_scrubber_cols.py
