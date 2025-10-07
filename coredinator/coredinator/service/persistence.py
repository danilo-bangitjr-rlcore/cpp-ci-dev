"""Service and bundle persistence layers for ServiceManager."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import backoff

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID, ServiceIntendedState, ServiceLike
from coredinator.services.registry import create_service_instance


class ServicePersistenceLayer:
    """Handles persistence of service state and configuration."""

    def __init__(self, db_path: Path):
        self._logger = get_logger(__name__)
        self._db_path = db_path
        self._init_database()

    def _init_database(self, retries: int = 1) -> None:
        """Initialize the service persistence database."""
        self._logger.info(
            "Initializing service persistence database",
            db_path=str(self._db_path),
            retries_remaining=retries,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=1,
            on_backoff=self._on_database_init_backoff,
            on_giveup=self._on_database_init_giveup,
        )
        def _do_init_database():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                self._create_service_tables(cursor)
                conn.commit()

        _do_init_database()

    def _on_database_init_backoff(self, details: Any):
        """Handle database initialization backoff - remove corrupted database."""
        self._logger.warning(
            "Service database initialization failed",
            db_path=str(self._db_path),
            retries_remaining=details["tries"] - 1,
        )
        self._logger.info(
            "Removing corrupted service database and retrying",
            db_path=str(self._db_path),
        )
        self._db_path.unlink(missing_ok=True)

    def _on_database_init_giveup(self, details: Any):
        """Handle final database initialization failure."""
        self._logger.error(
            "Service database initialization failed permanently",
            db_path=str(self._db_path),
        )

    def _recover_database_and_log(self, service_id: str | None, operation: str) -> None:
        """Recover database after error and log the attempt."""
        if service_id:
            self._logger.warning(f"Failed to {operation}", service_id=service_id)
        else:
            self._logger.warning(f"Failed to {operation}")
        self._init_database()

    def _create_service_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create database tables for service persistence."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_states (
                service_id TEXT PRIMARY KEY,
                service_type TEXT NOT NULL,
                intended_state TEXT NOT NULL CHECK (
                    intended_state IN ('running', 'stopped')
                ),
                config_path TEXT NOT NULL,
                base_path TEXT NOT NULL,
                process_ids TEXT,
                service_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def persist_service(self, service: ServiceLike, base_path: Path | None = None) -> None:
        """Persist service state to database."""
        status = service.status()
        process_ids = service.get_process_ids()

        # Extract service configuration data
        service_type = type(service).__name__
        config_path = str(status.config_path) if status.config_path else ""
        base_path_str = str(base_path) if base_path else ""
        process_ids_json = json.dumps(process_ids)

        self._logger.debug(
            "Persisting service state",
            service_id=service.id,
            service_type=service_type,
            intended_state=status.intended_state,
            config_path=config_path,
            process_ids=process_ids,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(service.id, "persist service state"),
        )
        def _do_persist_service():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO service_states
                    (service_id, service_type, intended_state, config_path, base_path, process_ids, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (service.id, service_type, status.intended_state, config_path, base_path_str, process_ids_json),
                )
                conn.commit()

        _do_persist_service()

    def load_services(self, base_path: Path | None = None) -> list[ServiceLike]:
        """Load services from database and create service instances."""
        self._logger.info(
            "Loading services from persistence database",
            db_path=str(self._db_path),
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(None, "load services"),
            on_giveup=lambda details: self._logger.warning("Failed to load services from database after recovery"),
        )
        def _do_load_services():
            services = []
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT service_id, service_type, intended_state, config_path, base_path, process_ids
                    FROM service_states
                    """,
                )

                for (
                    service_id, service_type, intended_state,
                    config_path, base_path_str, process_ids_json,
                ) in cursor.fetchall():
                    service_id = ServiceID(service_id)
                    config_path = Path(config_path)
                    loaded_base_path = Path(base_path_str) if base_path_str else (base_path or Path.cwd())
                    process_ids = json.loads(process_ids_json) if process_ids_json else []

                    self._logger.info(
                        "Loading service from database",
                        service_id=service_id,
                        service_type=service_type,
                        intended_state=intended_state,
                        config_path=str(config_path),
                        process_ids=process_ids,
                    )

                    # Create service instance based on type
                    maybe_service = create_service_instance(
                        service_id, service_type, config_path, loaded_base_path,
                    )
                    service = maybe_service.unwrap()
                    if service is None:
                        self._logger.warning(
                            "Failed to create service instance",
                            service_id=service_id,
                            service_type=service_type,
                        )
                        continue

                    services.append(service)

                    # Attempt to reattach to existing processes if service was intended to be running
                    if intended_state == ServiceIntendedState.RUNNING and process_ids:
                        self._attempt_service_reattachment(service, process_ids)
            return services

        try:
            return _do_load_services()
        except sqlite3.DatabaseError:
            # Return empty list on database error - services will be recreated as needed
            return []

    def remove_service(self, service_id: ServiceID) -> None:
        """Remove service from persistence database."""
        self._logger.debug(
            "Removing service from database",
            service_id=service_id,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(service_id, "remove service"),
        )
        def _do_remove_service():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM service_states WHERE service_id = ?", (service_id,))
                conn.commit()

        _do_remove_service()

    def _attempt_service_reattachment(self, service: ServiceLike, process_ids: list[int | None]) -> None:
        """Attempt to reattach service to existing processes."""
        if not process_ids or all(pid is None for pid in process_ids):
            return

        self._logger.info(
            "Attempting to reattach service to existing processes",
            service_id=service.id,
            process_ids=process_ids,
        )

        # Try to reattach to the first valid process ID
        for pid in process_ids:
            if pid is not None:
                success = service.reattach_process(pid)
                if success:
                    self._logger.info(
                        "Successfully reattached service to process",
                        service_id=service.id,
                        pid=pid,
                    )
                    return
                self._logger.debug(
                    "Failed to reattach service to process",
                    service_id=service.id,
                    pid=pid,
                )

        self._logger.info(
            "Process reattachment failed, service will need to be restarted",
            service_id=service.id,
        )
