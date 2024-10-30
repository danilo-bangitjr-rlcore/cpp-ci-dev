import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsx_to_csv as x2c

formal_sensor_names = {
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
    "FI-0887": "Scrubber Softened Water Flow PV",
    "FI-0887 (Estimated Setpoint)": "Scrubber Softened Water Flow SP",
    "FIC-3734 (PID Output)": "Scrubber Softened Water Flow SP",
    "FI-0872": "Scrubber Recirculation Flow",
    "AI-0879A": "Inlet H2S",
    "AI-0879B": "Outlet H2S",
    "Efficiency": "Efficiency",
    "TI-0880": "Inlet Temperature",
    "AI-0897B": "Softened Water Conductivity",
    "FV-3735": " Mist Eliminator Valve Status",
    "M-3739": "Blower Status",
    "Bleach_cost": "Bleach Cost",
    "Caustic_cost": "Caustic Cost",

    "ARC-3730 (PID External Setpoint)": "RLCORE pH SP",
    "ARC-3731 (PID External Setpoint)": "RLCORE ORP SP",
    "PDRC-3738 (PID External Setpoint)": "RLCORE DP",
    "FRC-3734 (External Setpoint)": "RLCORE Flow Rate"
}

def plot_columns(data, columns, highlight=None, label_interval=144, hline=None, aggregate=None):
    def plot_single_column(ax, data, ts, col, hl_idx, color):
        if col not in data.keys():
            return
        data0 = data[col]

        if aggregate is None:
            ln = ax.plot(data0, alpha=0.5, label=col, color=color)
            ax.scatter(hl_idx, data0[hl_idx], s=5, color='black')
            idx = np.arange(0, len(ts), label_interval)  # len(ts)//10)
            xtick = ts[idx]
        else:
            data_ag, _, xtick = aggregate_data(data0, ts, aggregate)
            ln = ax.plot(data_ag, alpha=1, label=col, color=color)
            idx = np.arange(0, len(xtick), label_interval)  # len(ts)//10)
            xtick = xtick[idx]
        return ln, idx, xtick


    fig, axs = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 3*len(columns)), sharex=True)
    if len(columns) == 1:
        axs = [axs]
    ts = data['Timestamp']
    if highlight is not None:
        hl_idx = dp.cut_timestamp_daily(ts, cut_hour=highlight)
    else:
        hl_idx = []
    for ci, col in enumerate(columns):
        if type(col) == str:
            ln, idx, xtick = plot_single_column(axs[ci], data, ts, col, hl_idx, color="C0")
            axs[ci].set_title(formal_sensor_names[col])
            axs[ci].legend()
        elif type(col) == list:
            lns = []
            for i, c in enumerate(col):
                # ax = axs[ci] if i==0 else axs[ci].twinx()
                ax = axs[ci] if i < 2 else axs[ci].twinx()
                ln, idx, xtick = plot_single_column(ax, data, ts, c, hl_idx, color='C{}'.format(i))
                lns += ln
                ax.tick_params(axis='y', colors='C{}'.format(i))
            labs = [l.get_label() for l in lns]
            axs[ci].legend(lns, labs, loc=0)
            axs[ci].set_title(formal_sensor_names[col[0]])
        else:
            raise NotImplementedError
        axs[ci].set_xticks(idx, labels=['']*len(idx))
        axs[ci].set_xticks(idx, labels=xtick, rotation=30, fontsize=8)

        if hline is not None:
            axs[ci].axhline(y=hline, linestyle='--', color='grey', linewidth=1, zorder=0)
    plt.tight_layout()
    plt.show()
    # plt.savefig(plot_pth, dpi=300)
    # print("Save plot {}".format(plot_pth))

def correlation_with_efficiency_3d(data, sensors, condition, clips):
    condition_data = data[condition[0]]
    condition_idx = np.where(np.logical_and(condition_data > condition[1][0], condition_data < condition[1][1]))[0]

    eff = data['Efficiency'].clip(0.9, 1)
    ax = plt.figure().add_subplot(projection='3d')
    x = data[sensors[0]].clip(clips[sensors[0]][0], clips[sensors[0]][1])
    y = data[sensors[1]].clip(clips[sensors[1]][0], clips[sensors[1]][1])
    z = data[sensors[2]].clip(clips[sensors[2]][0], clips[sensors[2]][1])

    # ax.plot_trisurf(x, y, z)
    p = ax.scatter(x, y, z, c='lightgray', s=3, alpha=0.)
    p = ax.scatter(x[condition_idx], y[condition_idx], z[condition_idx], c=eff[condition_idx], s=3, cmap='rainbow')
    # ax.scatter(x, y, z.min()-0.1*(z.max()-z.min()), c='grey', alpha=0.3)
    # ax.scatter(x, y.min()-0.1*(y.max()-y.min()), z, c='grey', alpha=0.3)
    # ax.scatter(x.min()-0.1*(x.max()-x.min()), y, z, c='grey', alpha=0.3)

    ax.set_xlabel(formal_sensor_names[sensors[0]])
    ax.set_ylabel(formal_sensor_names[sensors[1]])
    ax.set_zlabel(formal_sensor_names[sensors[2]])
    plt.colorbar(p)
    plt.tight_layout()
    plt.show()
    print("plt done")
    # plt.savefig(plot_pth, dpi=300)
    # print("Save plot {}".format(plot_pth))

def read_csv(filenames):
    date_col_name = "Timestamp"
    dfs = []
    for file in filenames:
        df = pd.read_csv(
            file,
            dtype=np.float32,
            skiprows=0,
            header=0,
            index_col=date_col_name,
            parse_dates=True,
        )
        dfs.append(df)

    concat_df = pd.concat(dfs)
    concat_df.sort_values(by=[date_col_name], inplace=True)
    concat_df = concat_df.ffill()
    concat_df["Timestamp"] = concat_df.index
    concat_df = concat_df.to_dict(orient="list")
    for k in concat_df:
        concat_df[k] = np.array(concat_df[k])
    return concat_df

def xlsx_to_csv_fn(filenames):
    dfs = []
    for f in filenames:
        print("Transferring {}.xlsx".format(f))
        df = x2c.raw2array("xlsx/{}.xlsx".format(f),
                           x2c.SENSORS+['ARC-3730 (PID External Setpoint)',
                                        'ARC-3731 (PID External Setpoint)',
                                        'PDRC-3738 (PID External Setpoint)',
                                        'FRC-3734 (External Setpoint)',
                                        'FIC-3734 (PID Process Value)',
                                        'FIC-3734 (PID Setpoint)',
                                        'FIC-3734 (PID Output)',
                                        ],
                           ignore_sensors=[])
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)
    # Add efficiency
    dfs["AI-0879B"] = dfs["AI-0879B"] / 1000.
    # dfs = x2c.scrubber_data_clean(dfs, outlier_window_length, save=True, clean_data_file="_new")
    dfs['Efficiency'] = x2c.h2s_efficiency(dfs)
    dfs["AI-0879B"] = dfs["AI-0879B"] * 1000. # get back to normal range

    # Add estimated flow rate setpoint
    dfs['FI-0887 (Estimated Setpoint)'] = x2c.estimate_setpoint(dfs['FI-0887'].to_numpy(),
                                                            [15, 20, 25, 30, 35, 40, 45, 50])
    return dfs

def estimate_chemical_usage(data):
    bleach_output = data["AIC-3731 (PID Output)"]
    bleach = bleach_output * 1.632 *0.83 / 12. # 5 min
    caustic_output = data["AIC-3730 (PID Output)"]
    caustic = caustic_output * 1.008 * 0.5412 / 12. # 5min
    data['Bleach_cost'] = bleach
    data['Caustic_cost'] = caustic
    return data

if __name__ == '__main__':
    # filenames = ["clean_data.csv"]
    # data = read_csv(filenames)
    # filenames = ["clean_data_test.csv"]
    # data = read_csv(filenames)

    filenames = ["Scrubber 4 DV Data Export Data Only Oct-21_Oct-27 2024"]
    data = xlsx_to_csv_fn(filenames)

    data = estimate_chemical_usage(data)

    plot_columns(data, [
        ['ARC-3730 (PID External Setpoint)', "AIC-3730 (PID Setpoint)"],
        ['ARC-3731 (PID External Setpoint)', "AIC-3731 (PID Setpoint)"],
        # ['PDRC-3738 (PID External Setpoint)', 'PDIC-3738 (PID Setpoint)'],
        ['FRC-3734 (External Setpoint)', 'FI-0887 (Estimated Setpoint)'],
        ["Efficiency"]
    ], label_interval=36)

    # plot_columns(data, [
    #     ["AIC-3730 (PID Setpoint)", "AIC-3730 (PID Process Value)", "AIC-3730 (PID Output)"],
    #     ["AIC-3731 (PID Setpoint)", "AIC-3731 (PID Process Value)", "AIC-3731 (PID Output)"],
    #     ["PDIC-3738 (PID Setpoint)", "PDIC-3738 (PID Process Value)", "PDIC-3738 (PID Output)"],
    #     ['FI-0887 (Estimated Setpoint)', 'FI-0887', 'FIC-3734 (PID Output)'],
    #     ["AI-0879A", "AI-0879B"],
    #     ["Efficiency"],
    # ], label_interval=36)
    #
    # plot_columns(data, [
    #     "PI-0169", "LI-3734 ", "FI-0872", "TI-0880", "AI-0897B",
    #     ["Efficiency"],
    #      ], label_interval=720)
    #
    plot_columns(data, [
        "AI-0897B",
        ["AI-0879A", "AI-0879B"],
        ["Efficiency"],
         ], label_interval=720)

