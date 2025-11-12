import time
from collections.abc import Iterable, Mapping

import numpy as np
import temain_mod  # pyright: ignore [reportMissingImports]
from asyncua.sync import Client, SyncNode
from asyncua.ua.uaerrors import BadNodeIdExists, BadNodeIdUnknown
from asyncua.ua.uatypes import VariantType
from corerl.config import MainConfig
from lib_config.loader import load_config

FOLDER_NAME = "TennesseeEastmanProcess"

STATES = [
    "XMEAS_1",  # A Feed  (stream 1)                    kscmh
    "XMEAS_2",  # D Feed  (stream 2)                    kg/hr
    "XMEAS_3",  # E Feed  (stream 3)                    kg/hr
    "XMEAS_4",  # A and C Feed  (stream 4)              kscmh (max 14)
    "XMEAS_5",  # Recycle Flow  (stream 8)              kscmh
    "XMEAS_6",  # Reactor Feed Rate  (stream 6)         kscmh
    "XMEAS_7",  # Reactor Pressure                      kPa gauge
    "XMEAS_8",  # Reactor Level                         %
    "XMEAS_9",  # Reactor Temperature                   Deg C
    "XMEAS_10", # Purge Rate (stream 9)                 kscmh
    "XMEAS_11", # Product Sep Temp                      Deg C
    "XMEAS_12", # Product Sep Level                     %
    "XMEAS_13", # Prod Sep Pressure                     kPa gauge
    "XMEAS_14", # Prod Sep Underflow (stream 10)        m3/hr
    "XMEAS_15", # Stripper Level                        %
    "XMEAS_16", # Stripper Pressure                     kPa gauge
    "XMEAS_17", # Stripper Underflow (stream 11)        m3/hr
    "XMEAS_18", # Stripper Temperature                  Deg C
    "XMEAS_19", # Stripper Steam Flow                   kg/hr
    "XMEAS_20", # Compressor Work                       kW
    "XMEAS_21", # Reactor Cooling Water Outlet Temp     Deg C
    "XMEAS_22", # Separator Cooling Water Outlet Temp   Deg C
    # Reactor Feed Analysis (Stream 6)
    #     Sampling Frequency = 0.1 hr
    #     Dead Time = 0.1 hr
    #     Mole %
    "XMEAS_23", # Component A
    "XMEAS_24", # Component B
    "XMEAS_25", # Component C
    "XMEAS_26", # Component D
    "XMEAS_27", # Component E
    "XMEAS_28", # Component F
    # Purge Gas Analysis (Stream 9)
    #     Sampling Frequency = 0.1 hr
    #     Dead Time = 0.1 hr
    #     Mole %
    "XMEAS_29", # Component A
    "XMEAS_30", # Component B
    "XMEAS_31", # Component C
    "XMEAS_32", # Component D
    "XMEAS_33", # Component E
    "XMEAS_34", # Component F
    "XMEAS_35", # Component G
    "XMEAS_36", # Component H
    # Product Analysis (Stream 11)
    #     Sampling Frequency = 0.25 hr
    #     Dead Time = 0.25 hr
    #     Mole %
    "XMEAS_37", # Component D
    "XMEAS_38", # Component E
    "XMEAS_39", # Component F
    "XMEAS_40", # Component G
    "XMEAS_41", # Component H
    #   Manipulated Variables
    "XMV_1",    # D Feed Flow (stream 2)            (Corrected Order)
    "XMV_2",    # E Feed Flow (stream 3)            (Corrected Order)
    "XMV_3",    # A Feed Flow (stream 1)            (Corrected Order)
    "XMV_4",    # A and C Feed Flow (stream 4)
    "XMV_5",    # Compressor Recycle Valve
    "XMV_6",    # Purge Valve (stream 9)
    "XMV_7",    # Separator Pot Liquid Flow (stream 10)
    "XMV_8",    # Stripper Liquid Product Flow (stream 11)
    "XMV_9",    # Stripper Steam Valve
    "XMV_10",   # Reactor Cooling Water Flow
    "XMV_11",   # Condenser Cooling Water Flow
    "XMV_12",    # Agitator Speed
]

ACTIONS = [
    "ACTIONS_1",  # Recycle flow: XMEAS(5)
    "ACTIONS_2",  # Product sep level: XMEAS(12)
    "ACTIONS_3",  # Stripper level: XMEAS(15)
    "ACTIONS_4",  # Stripper underflow: XMEAS(17)
    "ACTIONS_5",  # Stream 6 A: XMEAS(23)
    "ACTIONS_6", # Stream 6 D: XMEAS(26)
    "ACTIONS_7", # Stream 6 E: XMEAS(27)
    "ACTIONS_8",  # Stripper temperature: XMEAS(18)
    "ACTIONS_9",  # Reactor level: XMEAS(8)
    "ACTIONS_10",  # Reactor temperature: XMEAS(9)
    "ACTIONS_11", # Stream 9 B: XMEAS(30)
    "ACTIONS_12", # Stream 11 E: XMEAS(38)
    "ACTIONS_13", # Prod sep pressure: XMEAS(13)
]

AI_ACTIONS = [
    "AI_ACTIONS_1",  # Recycle flow: XMEAS(5)
    "AI_ACTIONS_2",  # Product sep level: XMEAS(12)
    "AI_ACTIONS_3",  # Stripper level: XMEAS(15)
    "AI_ACTIONS_4",  # Stripper underflow: XMEAS(17)
    "AI_ACTIONS_5",  # Stream 6 A: XMEAS(23)
    "AI_ACTIONS_6", # Stream 6 D: XMEAS(26)
    "AI_ACTIONS_7", # Stream 6 E: XMEAS(27)
    "AI_ACTIONS_8",  # Stripper temperature: XMEAS(18)
    "AI_ACTIONS_9",  # Reactor level: XMEAS(8)
    "AI_ACTIONS_10",  # Reactor temperature: XMEAS(9)
    "AI_ACTIONS_11", # Stream 9 B: XMEAS(30)
    "AI_ACTIONS_12", # Stream 11 E: XMEAS(38)
    "AI_ACTIONS_13", # Prod sep pressure: XMEAS(13)
]

MANUAL_ACTIONS = [
    "MANUAL_ACTIONS_1",  # Recycle flow: XMEAS(5)
    "MANUAL_ACTIONS_2",  # Product sep level: XMEAS(12)
    "MANUAL_ACTIONS_3",  # Stripper level: XMEAS(15)
    "MANUAL_ACTIONS_4",  # Stripper underflow: XMEAS(17)
    "MANUAL_ACTIONS_5",  # Stream 6 A: XMEAS(23)
    "MANUAL_ACTIONS_6", # Stream 6 D: XMEAS(26)
    "MANUAL_ACTIONS_7", # Stream 6 E: XMEAS(27)
    "MANUAL_ACTIONS_8",  # Stripper temperature: XMEAS(18)
    "MANUAL_ACTIONS_9",  # Reactor level: XMEAS(8)
    "MANUAL_ACTIONS_10",  # Reactor temperature: XMEAS(9)
    "MANUAL_ACTIONS_11", # Stream 9 B: XMEAS(30)
    "MANUAL_ACTIONS_12", # Stream 11 E: XMEAS(38)
    "MANUAL_ACTIONS_13", # Prod sep pressure: XMEAS(13)
]

DISTURBANCES = [
    "IDV_1", # A/C feed ratio B composition constant (stream 4),Step
    "IDV_2", # B composition A/C ratio constant (stream 4),Step
    "IDV_3", # D feed temperature (stream 2),Step
    "IDV_4", # Reactor cooling water inlet temperature,Step
    "IDV_5", # Condenser cooling water inlet temperature,Step
    "IDV_6", # A feed loss (stream 1),Step
    "IDV_7", # C header pressure loss—reduced availability (stream 4),Step
    "IDV_8", # "A, B, C feed composition (stream 4)",Random variation
    "IDV_9", # D feed temperature (stream 2),Random variation
    "IDV_10", # C feed temperature (stream 4),Random variation
    "IDV_11", # Reactor cooling water inlet temperature,Random variation
    "IDV_12", # Condenser cooling water inlet temperature,Random variation
    "IDV_13", # Reaction kinetics,Slow drift
    "IDV_14", # Reactor cooling water valve,Sticking
    "IDV_15", # Condenser cooling water valve,Sticking
    "IDV_16", # Unknown,Unknown
    "IDV_17", # Unknown,Unknown
    "IDV_18", # Unknown,Unknown
    "IDV_19", # Unknown,Unknown
    "IDV_20", # Unknown,Unknown
]


META = [
    "SIM_TIME",
    "MODE",
]

INITIAL_VALUES: Mapping[str, float] = {
    "XMEAS_1": 0.249,
    "XMEAS_2": 3669.258,
    "XMEAS_3": 4501.008,
    "XMEAS_4": 9.45,
    "XMEAS_5": 27.321,
    "XMEAS_6": 42.458,
    "XMEAS_7": 2706.356,
    "XMEAS_8": 74.639,
    "XMEAS_9": 120.396,
    "XMEAS_10": 0.336,
    "XMEAS_11": 80.221,
    "XMEAS_12": 48.711,
    "XMEAS_13": 2635.776,
    "XMEAS_14": 25.539,
    "XMEAS_15": 50.434,
    "XMEAS_16": 3102.475,
    "XMEAS_17": 22.326,
    "XMEAS_18": 65.758,
    "XMEAS_19": 228.785,
    "XMEAS_20": 41.5,
    "XMEAS_21": 94.614,
    "XMEAS_22": 77.284,
    "XMEAS_23": 32.188,
    "XMEAS_24": 8.893,
    "XMEAS_25": 26.383,
    "XMEAS_26": 6.882,
    "XMEAS_27": 18.776,
    "XMEAS_28": 1.657,
    "XMEAS_29": 32.958,
    "XMEAS_30": 13.823,
    "XMEAS_31": 23.978,
    "XMEAS_32": 1.257,
    "XMEAS_33": 18.579,
    "XMEAS_34": 2.263,
    "XMEAS_35": 4.844,
    "XMEAS_36": 2.299,
    "XMEAS_37": 0.018,
    "XMEAS_38": 0.836,
    "XMEAS_39": 0.099,
    "XMEAS_40": 53.724,
    "XMEAS_41": 43.828,
    "XMV_1": 62.963,
    "XMV_2": 54.079,
    "XMV_3": 24.792,
    "XMV_4": 59.818,
    "XMV_5": 22.567,
    "XMV_6": 40.223,
    "XMV_7": 34.307,
    "XMV_8": 47.537,
    "XMV_9": 47.58,
    "XMV_10": 41.237,
    "XMV_11": 19.582,
    "XMV_12": 0.0,
    "ACTIONS_1": 26.902,
    "ACTIONS_2": 50.0,
    "ACTIONS_3": 50.0,
    "ACTIONS_4": 22.949,
    "ACTIONS_5": 32.188,
    "ACTIONS_6": 6.8820,
    "ACTIONS_7": 18.776,
    "ACTIONS_8": 65.731,
    "ACTIONS_9": 75.000,
    "ACTIONS_10": 120.40,
    "ACTIONS_11": 13.823,
    "ACTIONS_12": 0.83570,
    "ACTIONS_13": 2633.7,
    "AI_ACTIONS_1": 26.902,
    "AI_ACTIONS_2": 50.0,
    "AI_ACTIONS_3": 50.0,
    "AI_ACTIONS_4": 22.949,
    "AI_ACTIONS_5": 32.188,
    "AI_ACTIONS_6": 6.8820,
    "AI_ACTIONS_7": 18.776,
    "AI_ACTIONS_8": 65.731,
    "AI_ACTIONS_9": 75.000,
    "AI_ACTIONS_10": 120.40,
    "AI_ACTIONS_11": 13.823,
    "AI_ACTIONS_12": 0.83570,
    "AI_ACTIONS_13": 2633.7,
    "MANUAL_ACTIONS_1": 26.902,
    "MANUAL_ACTIONS_2": 50.0,
    "MANUAL_ACTIONS_3": 50.0,
    "MANUAL_ACTIONS_4": 22.949,
    "MANUAL_ACTIONS_5": 32.188,
    "MANUAL_ACTIONS_6": 6.8820,
    "MANUAL_ACTIONS_7": 18.776,
    "MANUAL_ACTIONS_8": 65.731,
    "MANUAL_ACTIONS_9": 75.000,
    "MANUAL_ACTIONS_10": 120.40,
    "MANUAL_ACTIONS_11": 13.823,
    "MANUAL_ACTIONS_12": 0.83570,
    "MANUAL_ACTIONS_13": 2633.7,
    "IDV_1": 0.0,
    "IDV_2": 0.0,
    "IDV_3": 0.0,
    "IDV_4": 0.0,
    "IDV_5": 0.0,
    "IDV_6": 0.0,
    "IDV_7": 0.0,
    "IDV_8": 0.0,
    "IDV_9": 0.0,
    "IDV_10": 0.0,
    "IDV_11": 0.0,
    "IDV_12": 0.0,
    "IDV_13": 0.0,
    "IDV_14": 0.0,
    "IDV_15": 0.0,
    "IDV_16": 0.0,
    "IDV_17": 0.0,
    "IDV_18": 0.0,
    "IDV_19": 0.0,
    "IDV_20": 0.0,
    "SIM_TIME": 0.0,
    "MODE": 0.0,
}

ALL_NODE_NAMES = STATES + ACTIONS + MANUAL_ACTIONS + DISTURBANCES + META + AI_ACTIONS

def make_node_id(name: str, ns: int = 2):
    return f"ns={ns};s={name}"

def ensure_opc_folder(client: Client):
    folder_node_id = make_node_id(FOLDER_NAME)
    try:
        folder = client.nodes.objects.add_folder(folder_node_id, FOLDER_NAME)
    except BadNodeIdExists:
        # folder already exists
        folder = client.get_node(folder_node_id)
    return folder

def ensure_node_exists(client: Client, folder: SyncNode, name: str):
    node_identifier = make_node_id(name)

    node = client.get_node(node_identifier)
    val =  INITIAL_VALUES[name]

    try:
        _ = node.read_browse_name()
    except BadNodeIdUnknown:

        # node does not exist in OPC server, create it
        # instantiate first action as random sample, store in OPC
        var_type = VariantType.Double
        node = folder.add_variable(node_identifier, name, val, var_type)

    client.write_values([node], [val])
    return node

def ensure_opc_nodes(client: Client, node_names: list[str]):
    folder = ensure_opc_folder(client)
    opc_nodes: dict[str, SyncNode] = {}

    for name in node_names:
        node = ensure_node_exists(client, folder, name)
        opc_nodes[name] = node
    return opc_nodes

@load_config(MainConfig)
def main(cfg: MainConfig):
    opc_uri = cfg.coreio.opc_connections[0].opc_conn_url

    client = Client(opc_uri)

    opc_nodes = {}
    try:
        client.connect()
        opc_nodes = ensure_opc_nodes(client, ALL_NODE_NAMES)
    except Exception as e:
        print(e)

    state_nodes: Iterable[SyncNode] = [opc_nodes[name] for name in STATES]
    action_nodes: Iterable[SyncNode] = [opc_nodes[name] for name in ACTIONS]
    ai_action_nodes: Iterable[SyncNode] = [opc_nodes[name] for name in AI_ACTIONS]
    manual_action_nodes: Iterable[SyncNode] = [opc_nodes[name] for name in MANUAL_ACTIONS]
    disturbance_nodes: Iterable[SyncNode] = [opc_nodes[name] for name in DISTURBANCES]

    i = 0

    action_values_temp = None

    action_values_target = None
    action_values = None
    action_offsets = np.zeros(len(ACTIONS))

    disturbance_values = None

    def get_action(state: np.ndarray):
        nonlocal i, action_values, action_values_target, disturbance_values, action_values_temp, action_offsets
        if i%10 == 0:
            print('step')
            assert state.shape == (len(STATES),)

            # Write meta to OPC: right now META = ["SIM_TIME"]
            client.write_values([opc_nodes["SIM_TIME"]], [float(i*180)])

            # write state from fortran to OPC
            client.write_values(state_nodes, state)
            # read action from OPC and return to fortran

            mode = int(client.read_values([opc_nodes["MODE"]])[0])
            if mode == 0:
                print("Manual", end=" ")
                action_values_temp = np.asarray(client.read_values(manual_action_nodes))
            elif mode == 1:
                print("AI Control", end=" ")
                action_values_temp = np.asarray(client.read_values(ai_action_nodes))
            elif mode == 2:
                print('Operator', end=" ")
                # Pick random action values every 10th step
                action_values_temp = np.asarray(client.read_values(manual_action_nodes))
                if i%100 == 0:
                    action_offsets = np.random.randn(len(ACTIONS))*0.02*action_values_temp
                action_values_temp += action_offsets
                action_values_temp = np.clip(action_values_temp, 0, None)
                    # client.write_values(manual_action_nodes, action_values_temp)
            else:
                print("Unknown mode, falling back to manual", end=" ")
                action_values_temp = np.asarray(client.read_values(manual_action_nodes))

            client.write_values(action_nodes, action_values_temp)
            action_values_target = np.asarray(client.read_values(action_nodes))

            assert action_values_target.shape == (len(ACTIONS),)

            if action_values is None:
                action_values = action_values_target.copy()

            disturbance_values = np.array(client.read_values(disturbance_nodes),dtype=np.int64)
            assert disturbance_values.shape == (len(DISTURBANCES),)

            time.sleep(5)
        i += 1

        assert action_values_target is not None
        assert action_values is not None
        assert disturbance_values is not None

        action_values = (0.99)*action_values + (1-0.99)*action_values_target

        return action_values, disturbance_values


    # start simulation
    temain_mod.temain(get_action, np.int64(1))
    client.disconnect()



if __name__ == "__main__":
    main()
