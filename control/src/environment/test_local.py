# %%
from opc_connection import OpcConnection
from InfluxOPCEnv import InfluxOPCEnv, DBClientWrapper
from ReseauEnv import date_to_timestamp_reseau
import asyncio

import json

async def main():
    # %%
    db_settings_pth = "/Users/revanmacqueen/RL-Core/root/control/src/environment/reseau/db_settings_local.json"
    db_settings = json.load(open(db_settings_pth, "r"))
    opc_settings_pth = "/Users/revanmacqueen/RL-Core/root/control/src/environment/reseau/opc_settings_local.json"
    opc_settings = json.load(open(opc_settings_pth, "r"))
    


    # %%
    db_client = DBClientWrapper(db_settings["bucket"], db_settings["org"], 
                                db_settings["token"], db_settings["url"], date_fn=date_to_timestamp_reseau)

    # %%
    opc_connection = OpcConnection(opc_settings["IP"], opc_settings["port"])

    # %%
    FPM_control_tag = "osoyoos.plc.Process_DB.P250 Flow Pace Calc.Flow Pace Multiplier"
    data_folder = "/Users/revanmacqueen/RL-Core/root/control/src/environment/reseau/datalogs/"

    date_col = "Date "
    col_names = [" AIT101 NTU ",
                " AIT301 ppm ",
                " AIT401 NTU ",
                " FIT101 Lpm ",
                " FIT210 Lpm ",
                " FIT230 Lpm ",
                " FIT250 Lpm ",
                " FIT401 Lpm ", 
                " PT100 psi ", 
                " PT101 psi ", 
                " PT151 psi ", 
                " PT161 psi ", 
                " PT171 psi ", 
                " PT301 psi ", 
                " PT302 psi"]


    env = InfluxOPCEnv(db_client, opc_connection, 
                        runtime=10, 
                        control_tags=[FPM_control_tag], 
                        date_col=date_col, col_names=col_names,
                        offline_data_folder=data_folder)

    # %%
    env.reset()
    state, reward, done, _, _ = env.get_observations_since_time(0, 0)
    print("Success getting obs!")

    # %%
    await env.take_action(50)


asyncio.run(main())
print("Success taking action!")