import asyncio
import sys

from asyncua import Client, Server, ua


async def start():
    client = Client(url="opc.tcp://ignition-services:62541")
    client.application_uri = "urn:coag-workaround-client"

    server = Server()
    await server.init()
    await server.set_application_uri("urn:coag-workaround-server")
    server.set_endpoint("opc.tcp://0.0.0.0:50506")

    idx = await server.register_namespace("urn:coag-workaround-server")
    # this is the "root" node of the whole server
    objects = server.nodes.objects

    # this is the root node of our namespace
    rlcore_root = await objects.add_object(idx, "RLCore")

    dep_dose_sp_node = await rlcore_root.add_variable(
        "ns=1;s=dep_dose_sp", "Deployed PACl Dose Setpoint", ua.Variant(15, ua.VariantType.Double)
    )

    await client.connect()
    op_dose_sp_node = client.get_node("ns=1;s=[main-veolia]Dose_AL_SP_F.OUT")
    ai_dose_sp_node = client.get_node("ns=1;s=[main-veolia]AI_DOSE_SP")
    op_en_node = client.get_node("ns=1;s=[main-veolia]OPER_DOSE_CTRL_EN")
    ai_en_node = client.get_node("ns=1;s=[main-veolia]AI_DOSE_CTRL_EN")

    await server.start()

    try:
        while True:
            await asyncio.sleep(1)
            op_en, ai_en = await client.read_values([op_en_node, ai_en_node])

            # record the appropriate active setpoints for the agent
            # so it knows which set is properly active at any time

            # if operator mode is enabled, record the operator setpoints
            if op_en:
                op_val = await op_dose_sp_node.read_value()
                await dep_dose_sp_node.write_value(op_val)

            # if ai mode is enabled, record the agent setpoints
            if ai_en:
                ai_val = await ai_dose_sp_node.read_value()
                await dep_dose_sp_node.write_value(float(ai_val))

    except OSError as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass

    await client.disconnect()
    await server.stop()
    sys.exit(0)


asyncio.run(start())
