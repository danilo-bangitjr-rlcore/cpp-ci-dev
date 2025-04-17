import sys
import asyncio
from asyncua import Client, Server, ua

async def start():
    client = Client(url='opc.tcp://ignition-services:62541')
    client.application_uri = 'urn:backwash-workaround-client'

    server = Server()
    await server.init()
    await server.set_application_uri("urn:backwash-workaround-server")
    server.set_endpoint("opc.tcp://0.0.0.0:50505")

    idx = await server.register_namespace("urn:backwash-workaround-server")
    # this is the "root" node of the whole server
    objects = server.nodes.objects

    # this is the root node of our namespace
    rlcore_root = await objects.add_object(idx, "RLCore")

    ai_dur_sp_s = await rlcore_root.add_variable("ns=1;s=ai_dur_sp", "Agent Backwash Duration Setpoint", ua.Variant(15, ua.VariantType.Double))
    await ai_dur_sp_s.set_writable()

    ai_flow_sp_s = await rlcore_root.add_variable("ns=1;s=ai_flow_sp", "Agent Backwash Flow Setpoint", ua.Variant(170, ua.VariantType.Double))
    await ai_flow_sp_s.set_writable()

    dep_dur_sp_s = await rlcore_root.add_variable("ns=1;s=dep_dur_sp", "Deployed Backwash Duration Setpoint", ua.Variant(15, ua.VariantType.Double))
    dep_flow_sp_s = await rlcore_root.add_variable("ns=1;s=dep_flow_sp", "Deployed Backwash Flow Setpoint", ua.Variant(176, ua.VariantType.Double))

    await client.connect()
    uf2_mode_n = client.get_node('ns=1;s=[main-veolia]UF2_SEQ.STATE')
    op_dur_sp_n = client.get_node('ns=1;s=[main-veolia]BW_ST3_DUR_SP.OUT')
    op_flow_sp_n = client.get_node('ns=1;s=[main-veolia]BW_BP_FLOW_SP.OUT')
    ai_duration_sp_n = client.get_node('ns=1;s=[main-veolia]AI_BP_DUR_SP')
    ai_flow_sp_n = client.get_node('ns=1;s=[main-veolia]AI_BP_FLOW_SP')
    op_en_n = client.get_node('ns=1;s=[main-veolia]OPER_BW_CTRL_EN')
    ai_en_n = client.get_node('ns=1;s=[main-veolia]AI_BW_CTRL_EN')

    await server.start()

    try:
        vals = await client.read_values([uf2_mode_n])
        old_uf2_mode = vals[0]
        timer = 0
        timer_en = False
        while True:
            await asyncio.sleep(1)
            vals = await client.read_values([uf2_mode_n, op_dur_sp_n, op_flow_sp_n, op_en_n, ai_en_n])

            new_uf2_mode = vals[0]
            op_dur_sp = vals[1]
            op_flow_sp = vals[2]
            op_en = vals[3]
            ai_en = vals[4]

            # modes:
            # 16 - permeate down
            # 32 - backwash

            # when entering permeate down mode, write the agent setpoints
            # into the PLC. do this only once
            # also start the timer
            if old_uf2_mode != 16 and new_uf2_mode == 16:
                temp_dur = await ai_dur_sp_s.read_value()
                ai_duration_sp_dv = ua.DataValue(ua.Variant(temp_dur, ua.VariantType.Float))

                temp_flow = await ai_flow_sp_s.read_value()
                ai_flow_sp_dv = ua.DataValue(ua.Variant(temp_flow, ua.VariantType.Float))

                await client.write_values([ai_duration_sp_n, ai_flow_sp_n], [ai_duration_sp_dv, ai_flow_sp_dv])

                timer_en = True

            # when leaving backwash mode, write the operator setpoints
            # into the PLCs AI setpoints
            # also stop and reset the timer
            if old_uf2_mode == 32 and new_uf2_mode != 32:
                ai_duration_sp_dv = ua.DataValue(ua.Variant(op_dur_sp, ua.VariantType.Float))
                ai_flow_sp_dv = ua.DataValue(ua.Variant(op_flow_sp, ua.VariantType.Float))

                await client.write_values([ai_duration_sp_n, ai_flow_sp_n], [ai_duration_sp_dv, ai_flow_sp_dv])

                timer = 0
                timer_en = False

            # if the timer is enabled
            # if the timer has exceeded its duration, write the operator
            # setpoints into the PLCs AI setpoints and reset and disable the timer
            # otherwise increment the timer
            if timer_en == True:
                if timer > 300:
                    ai_duration_sp_dv = ua.DataValue(ua.Variant(op_dur_sp, ua.VariantType.Float))
                    ai_flow_sp_dv = ua.DataValue(ua.Variant(op_flow_sp, ua.VariantType.Float))

                    await client.write_values([ai_duration_sp_n, ai_flow_sp_n], [ai_duration_sp_dv, ai_flow_sp_dv])

                    timer = 0
                    timer_en = False
                else:
                    timer += 1


            old_uf2_mode = new_uf2_mode

            # record the appropriate active setpoints for the agent
            # so it knows which set is properly active at any time

            # if operator mode is enabled, recorded the operator setpoints
            if op_en == True:
                await dep_dur_sp_s.write_value(op_dur_sp)
                await dep_flow_sp_s.write_value(op_flow_sp)

            # if ai mode is enabled, record the agent setpoints
            if ai_en == True:
                temp_dur = await ai_dur_sp_s.read_value()
                temp_flow = await ai_flow_sp_s.read_value()
                await dep_dur_sp_s.write_value(float(temp_dur))
                await dep_flow_sp_s.write_value(float(temp_flow))


    except OSError as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


    await client.disconnect()
    await server.stop()
    sys.exit(0)


asyncio.run(start())
