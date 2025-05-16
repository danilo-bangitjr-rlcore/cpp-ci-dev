#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging

import numpy as np

from coreio.utils.zmq_communication import ZMQ_Communication
from coreio.utils.opc_communication import OPC_Communication
from corerl.config import MainConfig
from corerl.configs.loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@load_config(MainConfig, base='config/')
async def main(cfg: MainConfig):
    opc_communication = await OPC_Communication().init(cfg.coreio, cfg.pipeline.tags)
    zmq_communication = ZMQ_Communication(cfg.coreio)
    
    zmq_communication.start()
    await opc_communication.start()

    for i in range(20):
        event = zmq_communication.recv_event()
        if event is None:
            print(f"{i}: no event received (timeout)")
            continue

        print(f"{i}: event received")
        print(event)
        

    # zmq_communication.listen_forever()
        # for action_name in opc_communication.action_nodes.keys():
            # opc_communication.action_nodes[action_name].value = float(np.random.randint(0, 11))
            # print(action_name, opc_communication.action_nodes[action_name])

        # await opc_communication.emit_action(opc_communication.action_nodes)

    zmq_communication.cleanup()
    await opc_communication.cleanup()
    print("Finished")

if __name__ == "__main__":
    asyncio.run(main())
