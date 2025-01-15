#!/usr/bin/env python
import asyncio
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from asyncua import Server


async def main(args: Namespace):
    _logger = logging.getLogger(__name__)

    # Setup server
    server = Server()
    await server.init()

    server.set_endpoint(args.url)
    server.set_server_name(args.name)
    server.disable_clock(args.disable_clock)

    if args.certificate:
        await server.load_certificate(args.certificate)
    if args.private_key:
        await server.load_private_key(args.private_key)

    await server.set_application_uri(args.uri)
    idx = await server.register_namespace(args.namespace)

    # this is the root node of our namespace
    node_obj = await server.nodes.objects.add_object(idx, "SimulationRoot")
    node_var = await node_obj.add_variable(idx, "Counter", 0.0)

    # Set MyVariable to be writable by clients
    await node_var.set_writable()

    _logger.info("Starting server!")
    async with server:
        while True:
            await asyncio.sleep(1)
            new_val = await node_var.get_value() + 0.1
            _logger.debug("Set value of %s to %.1f", node_var, new_val)
            await node_var.write_value(new_val)


if __name__ == "__main__":
    parser = ArgumentParser(description="", formatter_class=ArgumentDefaultsHelpFormatter)
    # we set up a server, this is a bit different from other tool, so we do not reuse common arguments
    parser.add_argument(
        "-u",
        "--url",
        help="URL/endpoint of OPC UA server",
        default="opc.tcp://0.0.0.0:4840/rlcore/server/",
        metavar="URL",
    )
    parser.add_argument(
        "-n", "--name", help="Name of OPC UA server", default="RLCore OPC-UA E2E Server", metavar="NAME"
    )
    parser.add_argument(
        "-i",
        "--uri",
        help="Server URI, should be unique in system and prefixed with urn:",
        default="urn:rlcore:opc:e2e:server",
    )
    parser.add_argument(
        "-s", "--namespace", help="Namespace of OPC UA server", default="http://simulation.e2e.rlcore.ai"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set log level",
    )
    parser.add_argument(
        "-c",
        "--disable-clock",
        action="store_true",
        help="Disable clock, to avoid seeing many write if debugging an application",
    )
    parser.add_argument("--certificate", help="path to certificate, either .pem or .der")
    parser.add_argument("--private_key", help="path to private key, either .pem or .der")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    asyncio.run(main(args), debug=True)
