from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from coredinator.logging_config import get_logger
from coredinator.service.persistence import ServicePersistenceLayer
from coredinator.service.protocols import (
    ServiceBundle,
    ServiceBundleID,
    ServiceID,
    ServiceLike,
)


class ServiceManager:
    """Central manager for all service instances with ownership tracking.

    No other object should own service instances directly.
    Services are reference-counted by bundles that depend on them.
    """

    def __init__(self, base_path: Path):
        self._logger = get_logger(__name__)
        self._services: dict[ServiceID, ServiceLike] = {}
        # Track which bundles own which services (many-to-many relationship)
        self._service_owners: defaultdict[ServiceID, set[ServiceBundleID]] = defaultdict(set)
        self._bundle_services: defaultdict[ServiceBundleID, set[ServiceID]] = defaultdict(set)

        # Service persistence setup
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

        self._service_owners.pop(service_id, None)
        for bundle_services in self._bundle_services.values():
            bundle_services.discard(service_id)

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

    def register_bundle(self, bundle: ServiceBundle):
        """Register a service bundle and its service dependencies."""
        bundle_id = bundle.id
        required_services = bundle.get_required_services()

        self._logger.info(
            "Registering service bundle",
            bundle_id=bundle_id,
            required_services=list(required_services),
        )

        self._bundle_services[bundle_id] = required_services

        for service_id in required_services:
            self._service_owners[service_id].add(bundle_id)

            # Persist updated service for existing services
            service = self.get_service(service_id)
            if service is not None:
                self._service_persistence.persist_service(service, self._base_path)

    def unregister_bundle(self, bundle_id: ServiceBundleID, grace_seconds: float = 5.0):
        """Unregister a service bundle and stop services that no longer have owners.

        Returns list of services that were actually stopped.
        """
        self._logger.info(
            "Unregistering service bundle",
            bundle_id=bundle_id,
            grace_seconds=grace_seconds,
        )
        stopped_services: list[ServiceID] = []
        bundle_services = self._bundle_services.pop(bundle_id, set())

        for service_id in bundle_services:
            owners = self._service_owners[service_id]
            owners.discard(bundle_id)

            if owners:
                # Service still has owners, just update service in database
                service = self.get_service(service_id)
                if service is not None:
                    self._service_persistence.persist_service(service, self._base_path)
                continue

            service = self.get_service(service_id)
            if service is None:
                self._logger.warning(
                    "Service not found during cleanup",
                    service_id=service_id,
                )
                continue

            self._logger.info(
                "Stopping orphaned service",
                service_id=service_id,
                grace_seconds=grace_seconds,
            )
            service.stop(grace_seconds)
            stopped_services.append(service_id)

            # Persist updated service state after stopping
            self._service_persistence.persist_service(service, self._base_path)

        return stopped_services

    def get_service_owners(self, service_id: ServiceID):
        """Get all bundles that depend on this service."""
        return self._service_owners[service_id]

    def get_bundle_services(self, bundle_id: ServiceBundleID):
        """Get all services this bundle depends on."""
        return self._bundle_services[bundle_id]

    def is_service_shared(self, service_id: ServiceID):
        """Check if a service is shared by multiple bundles."""
        return len(self._service_owners[service_id]) > 1

    def can_stop_service(self, service_id: ServiceID):
        """Check if a service can be safely stopped (no dependent bundles)."""
        return len(self._service_owners[service_id]) == 0

    def add_service_owner(self, service_id: ServiceID, owner_id: ServiceBundleID):
        """Add an owner to a service for tracking purposes."""
        self._service_owners[service_id].add(owner_id)

    def remove_service_owner(self, service_id: ServiceID, owner_id: ServiceBundleID):
        """Remove an owner from a service."""
        self._service_owners[service_id].discard(owner_id)

    def update_service_state(self, service_id: ServiceID):
        """Update the persisted state of a service after lifecycle changes."""
        service = self.get_service(service_id)
        if service is not None:
            self._service_persistence.persist_service(service, self._base_path)
