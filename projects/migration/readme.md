# ATTENTION!!!!

This is an offline backup of the steps required to migrate from narrow to wide format in Drayton Valley.
To find the most updated version, please see [here](https://docs.google.com/document/d/1Eu56ZjQJxGkBm1e1zZy5jrdmoztU5ijQvOlRR9swj_k/edit?tab=t.0)

== SUMMARY OF STEPS ==

(1) Query tall table to get distinct column names and data types, outputs text file
(2) Generate create table statement with python script, outputs text file
(3) Create wide table
(4) Use sed script to get list of column names
(5) Insert list of column names into rust script
(6) Copy tall data into long text format
(7) Clean up col names in tall format text file
(8) Pivot tall format text file with rust script, outputs wide format txt file
(9) Copy wide txt file into wide SQL table
(10) Create hypertable (with migration) on newly filled wide table
======================

Get distinct sensor names and dtypes

First, ssh into the Drayton Valley machine.

```
docker exec -it timescaledb psql -U postgres -d postgres
```

Run this inside the container– takes 1-1.5 minutes to run.
```
\COPY (SELECT DISTINCT ON (id) id, any_value(process), any_value(name), jsonb_typeof(any_value(fields)->'val') as dtype FROM main GROUP BY id) TO /var/lib/postgresql/data/raw_id_datatype.txt
```

Now, in your local computer run:
```
cd projects/migration
```

```
scp dv-rl-wg:/rlcore/timescaledb/var/lib/data/raw_id_datatype.txt ./data 
```
Sometimes, we don't have permission to read from that DV dir, so copy it into your DV user's home dir.

Now use this sed script to clean the headers
```
sed -E -f utils/clean_header.sed data/raw_id_datatype.txt > data/main_wide_header.txt
```

Output cmd to create the wide table. Note, the venv for running python is in `/projects`, activate it first. 
```
python utils/create_wide.py --input-file data/main_wide_header.txt > data/create_table_stmt.txt
```

On the DV machine, you can use that command to create the wide table
```
psql -U postgres -p 5432 -h localhost -f /home/kerrick/drayton_valley/combined_wide/create_table_stmt.txt
```

Next, on local, strip out the data types
```
sed -E -f utils/get_col_names.sed data/create_table_stmt.txt > data/wide_col_names.txt
```

On the DV machine, copy long data from the old table. 
Don't forget to mount to the timescale docker using docker exec -it …
Pro tip: you can add `LIMIT 10` if just testing
```
\COPY (SELECT time, id, process, name, fields->>'val' as val FROM main ORDER BY time ASC) TO /var/lib/postgresql/data/main_long_id.txt
```

Now, same as before, use `scp` to copy the `main_long_id.txt` file into your local computer. 

```
# this will combine id, process, and name into a unique lower case tag name
sed -E -i -f utils/clean_header_with_timestamps.sed data/main_long_id.txt
```

Cd to cargo dir, streaming_pivot
```
cargo build --release

./target/release/streaming_pivot --input-fie ../data/main_long_id.txt --col-names ../data/wide_col_names.txt --output-file ../data/main_wide_id.txt
```

You can now upload `/data/main_wide_id.txt` to the DV machine
```
\COPY main_wide FROM /home/kerrick/drayton_valley/combined_wide/main_wide_id.txt
```

```
SELECT create_hypertable('main_wide', by_range('time'), migrate_data => true);
```
