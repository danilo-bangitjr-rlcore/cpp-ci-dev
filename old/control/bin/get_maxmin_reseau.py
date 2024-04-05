import os, sys
sys.path.insert(0, '..')

from src.environment.InfluxOPCEnv import InfluxOPCEnv, DBClientWrapperBase
import json
import time

os.chdir("..")
print("Change dir to", os.getcwd())
 
db_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\db_settings_osoyoos.json"
db_settings = json.load(open(db_settings_pth, "r"))

opc_settings_pth = "\\Users\\RLCORE\\root\\control\\src\\environment\\reseau\\opc_settings_osoyoos.json"
opc_settings = json.load(open(opc_settings_pth, "r"))

db_client = DBClientWrapperBase(db_settings["bucket"], db_settings["org"], 
                    db_settings["token"], db_settings["url"])

window_days = 20
window_seconds = 60*60*24*window_days
now = int(time.time())
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
        
df = db_client.query(now-window_seconds, now, col_names=col_names, include_time=False)
print(df.agg(['min', 'max']))