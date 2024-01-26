
from opc_connection import OpcConnection
from InfluxOPCEnv import InfluxOPCEnv, DBClientWrapper
from ReseauEnv import date_to_timestamp_reseau
from ReseauEnv import ReseauEnv

import json
import asyncio
import time
def main():
    db_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\db_settings_osoyoos.json"
    db_settings = json.load(open(db_settings_pth, "r"))
    
    opc_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\opc_settings_osoyoos.json"
    opc_settings = json.load(open(opc_settings_pth, "r"))
    
    db_client = DBClientWrapper(db_settings["bucket"], db_settings["org"], 
                        db_settings["token"], db_settings["url"])
    
    opc_connection = OpcConnection(opc_settings["IP"], opc_settings["port"])

    control_tags = ["osoyoos.plc.Process_DB.P250 Flow Pace Calc.Flow Pace Multiplier"]
    control_tag_default = [15]
    runtime = None
    date_col = "Date "
    col_names = [
        "ait101_pv",
        "ait301_pv",
        "ait401_pv",
        "fit101_pv",
        "fit210_pv",
        "fit230_pv",
        "fit250_pv",
        "fit401_pv", 
        "p250_fp", 
        "pt100_pv",
        "pt101_pv", 
        "pt161_pv"
        ]


    env = ReseauEnv(db_client, opc_connection,  control_tags, control_tag_default, date_col, col_names, runtime, decision_freq=600)
    s_0 = env.reset()

    state, reward, done, _, _ = env.get_observation(0)
    print(state)
    print("Success getting obs!")
    s = time.time()
    env.take_action([15])
    print(time.time()-s)
    s = time.time()
    env.take_action([15])
    print(time.time()-s)
    print("Success taking action!")
    
main()