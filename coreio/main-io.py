#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging

import numpy as np

from coreio.utils.opc_communication import OPC_Communication
from corerl.config import MainConfig
from corerl.configs.loader import load_config

logger = logging.getLogger(__name__)

@load_config(MainConfig, base='config/')
async def main(cfg: MainConfig):
    opc_communication = await OPC_Communication().init(cfg.coreio, cfg.pipeline.tags)

    async with opc_communication:
        for _ in range(10):
            for action_name in opc_communication.action_nodes.keys():
                opc_communication.action_nodes[action_name].value = float(np.random.randint(0, 11))
                print(action_name, opc_communication.action_nodes[action_name])

            await opc_communication.emit_action(opc_communication.action_nodes)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
