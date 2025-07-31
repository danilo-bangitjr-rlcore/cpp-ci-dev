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

```
\COPY (SELECT DISTINCT ON (id) id, any_value(process), any_value(name), jsonb_typeof(any_value(fields)->'val') as dtype FROM main GROUP BY id) TO /var/lib/postgresql/data/raw_id_datatype.txt
```
