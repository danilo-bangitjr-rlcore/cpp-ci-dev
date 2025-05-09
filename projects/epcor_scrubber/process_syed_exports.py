from pathlib import Path

import polars as pl

manual_test_file = Path(
    "projects/epcor_scrubber/syed_exports/testplan_data/Scrubber 4 DV Data Export Data Only Jun-24_Jul-01 2024.xlsx"
)
dfs = pl.read_excel(manual_test_file, sheet_id=0, has_header=False)


def merge_insane_header(df: pl.DataFrame):
    name_df = df[:3].transpose().fill_null(strategy="forward")
    new_col_names = pl.concat_str([name_df[:, 0], name_df[:, 1], name_df[:, 2]], separator=" ")
    name_df = name_df.with_columns(new_col_names.alias("new col names"))

    new_col_mapping = {df.columns[i]: name_df["new col names"][i] for i in range(len(df.columns))}
    df = df.rename(new_col_mapping)[3:]
    return df

def remove_unused_cols(df: pl.DataFrame):
    # remove status cols
    df = df.drop([col for col in df.columns if "Parameter Status" in col])

    # remove duplicate timestamp columns
    df = pl.concat(
        [df[:, 0].to_frame(), df[:, 1:].drop([col for col in df[:, 1:].columns if "Timestamp" in col])],
        how="horizontal",
    )
    return df
#####################
# H2S
#####################

h2s_df = dfs["H2S"]

h2s_df = merge_insane_header(h2s_df)
h2s_df = remove_unused_cols(h2s_df)

# rename columns
h2s_df = h2s_df.rename(lambda c: "time" if "Timestamp" in c else c)
h2s_df = h2s_df.rename(lambda c: "ai0879a" if "AI-0879A" in c else c)
h2s_df = h2s_df.rename(lambda c: "ai0879b" if "AI-0879B" in c else c)


#####################
# ORP pH controllers
#####################

orp_ph_df = dfs["ORP_pH Controllers"]

orp_ph_df = merge_insane_header(orp_ph_df)
orp_ph_df = remove_unused_cols(orp_ph_df)

# rename columns
orp_ph_df = orp_ph_df.rename(lambda c: "time" if "Timestamp" in c else c)

orp_ph_df = orp_ph_df.rename(lambda c: "aic3730_out" if "AIC-3730 (PID Output)" in c else c)
orp_ph_df = orp_ph_df.rename(lambda c: "aic3731_out" if "AIC-3731 (PID Output)" in c else c)

orp_ph_df = orp_ph_df.rename(lambda c: "aic3730_sp" if "AIC-3730 (PID Setpoint)" in c else c)
orp_ph_df = orp_ph_df.rename(lambda c: "aic3731_sp" if "AIC-3731 (PID Setpoint)" in c else c)

orp_ph_df = orp_ph_df.rename(lambda c: "aic3730_pv" if "AIC-3730 (PID Process Value)" in c else c)
orp_ph_df = orp_ph_df.rename(lambda c: "aic3731_pv" if "AIC-3731 (PID Process Value)" in c else c)


final_df = h2s_df.join(orp_ph_df, on="time")
print(final_df.head())

write_path = manual_test_file.parent / (manual_test_file.stem.replace(" ", "_") + "sane.csv")
final_df.write_csv(write_path)

final_df = pl.concat(
    [final_df[:, 0].to_frame(), final_df.select(pl.exclude("time").str.to_decimal()).cast(pl.Float64)], how="horizontal"
)
final_df = final_df.with_columns(
    (final_df.get_column("ai0879a") - final_df.get_column("ai0879b") / 1000).alias("h2s_removal")
)
final_df = final_df.with_columns(
    (final_df.get_column("h2s_removal") / final_df.get_column("ai0879a")).alias("efficiency")
)
final_df = final_df.with_columns(
    (final_df.get_column("aic3730_out") * 0.5455 + final_df.get_column("aic3731_out") * 1.355).alias("chem_cost")
)
final_df = final_df.with_columns(
    (final_df.get_column("chem_cost") / final_df.get_column("h2s_removal")).alias("cost_per_ppmhr")
)
print(final_df.head())
print(final_df.describe())
