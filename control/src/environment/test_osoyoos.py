
from opc_connection import OpcConnection
from InfluxOPCEnv import InfluxOPCEnv, DBClientWrapperBase
from ReseauEnv import date_to_timestamp_reseau
from ReseauEnv import ReseauEnv
from state_constructor import *

import json
import asyncio
import time
def main():
    db_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\db_settings_osoyoos.json"
    db_settings = json.load(open(db_settings_pth, "r"))
    
    opc_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\opc_settings_osoyoos.json"
    opc_settings = json.load(open(opc_settings_pth, "r"))
    
    db_client = DBClientWrapperBase(db_settings["bucket"], db_settings["org"], 
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




    env = ReseauEnv(db_client, opc_connection,  control_tags, control_tag_default, date_col, 
        col_names, runtime, obs_freq=1800, obs_window=1800, last_n_obs=1700 )
    s_0 = env.reset()

    # s1 = MaxminNormalize(env)
    # s2 = WindowAverage(5)
    # s2.set_parents([s1])
    # s3 = End()
    # s3.set_parents([s2])
    # s4 = MemoryTrace(0.9)
    # s4.set_parents([s3])
    # s5 = Concatenate()
    # s5.set_parents([s3, s4])
    # sc = StateConstructorWrapper(s5, time_frame=1700)

    s1 = MaxminNormalize(env)
    s2 = WindowAverage(3)
    s2.set_parents([s1])

    s3 = End()
    s3.set_parents([s2])

    s4 = KOrderHistory(1)
    s4.set_parents([s3])

    s5 = Flatten()
    s5.set_parents([s4])
    sc = StateConstructorWrapper(s5)



    for i in range(5):
        t = time.time()
        state, reward, done, _, _ = env.get_observation(0)
        print("Time to get state from db: {}".format(time.time()-t))
        t = time.time()
        state = sc(state)
        print(state.shape)
        print(state)


    # assert state.shape[0] == sc.get_state_dim(12)




    print("Time construct state: {}".format(time.time()-t))
    
main()