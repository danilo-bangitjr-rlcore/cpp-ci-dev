import asyncio
import logging
import threading
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lib_defs.type_defs.base_events import EventTopic, EventType
from service_framework.service import RLTuneService

from coreio.communication.opc_communication import OPC_Connection_IO
from coreio.communication.scheduler import start_scheduler_io_thread
from coreio.communication.sql_communication import SQL_Manager
from coreio.communication.zmq_communication import ZMQ_Communication
from coreio.utils.config_schemas import InteractionConfigAdapter, MainConfigAdapter, PipelineConfigAdapter
from coreio.utils.event_handlers import handle_read_event, handle_write_event
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType
from coreio.utils.opc_utils import concat_opc_nodes, initialize_opc_connections

logger = logging.getLogger(__name__)


class CoreIOService(RLTuneService):
    def __init__(self, cfg: MainConfigAdapter):
        super().__init__(
            service_name='coreio',
            event_topic=EventTopic.coreio,
            event_bus_host=cfg.coreio.event_bus_client.host,
            event_bus_pub_port=cfg.coreio.event_bus_client.pub_port,
            event_bus_sub_port=cfg.coreio.event_bus_client.sub_port,
            event_bus_enabled=cfg.coreio.event_bus_client.enabled,
        )
        self.cfg = cfg

        self.opc_connections: dict[str, OPC_Connection_IO] | None = None
        self.zmq_communication: ZMQ_Communication | None = None
        self.sql_communication: SQL_Manager | None = None
        self.ingress_stop_event: threading.Event | None = None
        self.all_registered_nodes = None

    async def _do_start(self):
        if self.cfg.coreio.log_file is not None:
            self._enable_log_files(Path(self.cfg.coreio.log_file))

        logger.info("Starting OPC Connections")

        if self.cfg.coreio.data_ingress.enabled:
            logger.debug("Will initialize OPC connections using coreio.tags (data_ingress enabled)")
            self.opc_connections = await initialize_opc_connections(
                self.cfg.coreio.opc_connections,
                self.cfg.coreio.tags,
            )
        else:
            logger.warning(
                "Running in compatibility mode, data ingress disabled. "
                "Will initialize OPC connections using pipeline.tags (data_ingress disabled). "
                "This feature will be deprecated soon",
            )

            assert isinstance(self.cfg.pipeline, PipelineConfigAdapter), "Please define tags under pipeline.tags"

            heartbeat = None
            if isinstance(self.cfg.interaction, InteractionConfigAdapter):
                heartbeat = self.cfg.interaction.heartbeat

            self.opc_connections = await initialize_opc_connections(
                self.cfg.coreio.opc_connections,
                self.cfg.pipeline.tags,
                heartbeat,
            )

        logger.info("Starting ZMQ communication")
        self.zmq_communication = ZMQ_Communication(
            event_class=IOEvent,
            topic=IOEventTopic.coreio,
            consumer_name="coreio_consumer",
            subscriber_addrs=[self.cfg.coreio.coreio_origin, self.cfg.coreio.coreio_app],
            publisher_addr=self.cfg.coreio.coreio_app,
        )

        logger.debug("Starting ZMQ communication thread/loop")
        self.zmq_communication.start()

        assert self.opc_connections is not None
        opc_connections = self.opc_connections

        logger.debug("Attaching write_to_opc callback")
        self.zmq_communication.attach_callback(
            event_type=IOEventType.write_to_opc,
            cb=lambda event: handle_write_event(event, opc_connections),
        )

        if self.cfg.coreio.data_ingress.enabled:
            logger.info("Starting SQL communication - Data ingress enabled")
            logger.debug(f"Concatenating nodes from {len(self.opc_connections)} OPC connections")
            self.all_registered_nodes = concat_opc_nodes(self.opc_connections, skip_heartbeat=True)
            logger.debug(f"Creating SQL Manager for {len(self.all_registered_nodes)} nodes")
            self.sql_communication = SQL_Manager(
                self.cfg.infra,
                table_name=self.cfg.env.db.table_name,
                nodes_to_persist=self.all_registered_nodes,
            )

            self.ingress_stop_event = threading.Event()
            logger.debug("About to start scheduler IO thread for data ingress")
            start_scheduler_io_thread(self.cfg.coreio.data_ingress, self.ingress_stop_event, self.zmq_communication)

            assert self.opc_connections is not None
            assert self.sql_communication is not None
            opc_connections = self.opc_connections
            sql_communication = self.sql_communication

            logger.debug("Attaching read_from_opc callback")
            self.zmq_communication.attach_callback(
                event_type=IOEventType.read_from_opc,
                cb=lambda event: handle_read_event(
                    event,
                    opc_connections,
                    sql_communication,
                    read_period=self.cfg.coreio.data_ingress.ingress_period,
                ),
            )

        logger.info("CoreIO is ready")

    async def _do_run(self):
        assert self.zmq_communication is not None

        logger.debug("Starting main event loop")
        async for event in self.zmq_communication.async_listen_forever():
            logger.debug(f"Processing event {event.type} in main loop")
            if event.type == EventType.exit_io:
                logger.info("Received exit event, shutting down CoreIO...")
                break

            await asyncio.sleep(0)

    async def _do_stop(self):
        if self.ingress_stop_event is not None:
            logger.debug("Stopping ingress thread")
            self.ingress_stop_event.set()

        if self.zmq_communication is not None:
            logger.debug("Closing ZMQ communication")
            self.zmq_communication.cleanup()

        if self.sql_communication is not None:
            logger.debug("SQL communication cleanup (no explicit cleanup needed)")

        if self.opc_connections is not None:
            logger.debug(f"Closing {len(self.opc_connections)} OPC connections")
            for conn in self.opc_connections.values():
                await conn.cleanup()

    def _enable_log_files(self, log_path: Path):
        save_path = log_path.parent / str(datetime.now(UTC).date())
        save_path.mkdir(exist_ok=True, parents=True)
        file_handler = RotatingFileHandler(
            filename=save_path / log_path.name,
            maxBytes=10_000_000,
            backupCount=3,
        )
        log_fmt = "[%(asctime)s][%(levelname)s] - %(message)s"
        file_handler.setFormatter(logging.Formatter(log_fmt))
        logging.getLogger().addHandler(file_handler)
