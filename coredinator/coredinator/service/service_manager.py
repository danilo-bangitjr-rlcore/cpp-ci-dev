from collections.abc import Callable
from pathlib import Path

from coredinator.logging_config import get_logger
from coredinator.service.persistence import ServicePersistenceLayer
from coredinator.service.protocols import ServiceID, ServiceLike


class ServiceManager:
    """Central manager for all service instances.

    No other object should own service instances directly.
    """

    def __init__(self, base_path: Path):
        self._logger = get_logger(__name__)
        self._services: dict[ServiceID, ServiceLike] = {}

        # Service persistence setup
        if base_path.exists() and not base_path.is_dir():
            raise ValueError(f"Base path must be a directory: {base_path}")

        base_path.mkdir(parents=True, exist_ok=True)

        self._base_path = base_path
        self._service_persistence = ServicePersistenceLayer(base_path / "service_state.db")

        # Load existing services from persistence
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        """Load services from persistence layer."""
        # Load services
        services = self._service_persistence.load_services(self._base_path)
        for service in services:
            self._services[service.id] = service

    def register_service(self, service: ServiceLike) -> None:
        self._logger.info(
            "Registering service",
            service_id=service.id,
            service_type=type(service).__name__,
        )
        self._services[service.id] = service
        self._service_persistence.persist_service(service, self._base_path)

    def get_service(self, service_id: ServiceID) -> ServiceLike | None:
        return self._services.get(service_id)

    def remove_service(self, service_id: ServiceID) -> ServiceLike | None:
        self._logger.info(
            "Removing service",
            service_id=service_id,
        )
        service = self._services.pop(service_id, None)
        if service is None:
            self._logger.warning(
                "Attempted to remove non-existent service",
                service_id=service_id,
            )
            return None

        self._service_persistence.remove_service(service_id)

        return service

    def list_services(self) -> list[ServiceID]:
        return list(self._services.keys())

    def has_service(self, service_id: ServiceID) -> bool:
        return service_id in self._services

    def get_or_register_service(self, service_id: ServiceID, service_factory: Callable[[], ServiceLike]) -> ServiceLike:
        existing_service = self.get_service(service_id)
        if existing_service is not None:
            return existing_service

        self._logger.info(
            "Creating new service via factory",
            service_id=service_id,
        )
        service = service_factory()
        self.register_service(service)
        return service

    def update_service_state(self, service_id: ServiceID):
        """Update the persisted state of a service after lifecycle changes."""
        service = self.get_service(service_id)
        if service is not None:
            self._service_persistence.persist_service(service, self._base_path)
