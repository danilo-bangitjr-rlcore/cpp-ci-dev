from collections.abc import Callable

from coredinator.service.protocols import ServiceID, ServiceLike


class ServiceManager:
    """Central manager for all service instances.

    No other object should own service instances directly.
    """

    def __init__(self):
        self._services: dict[ServiceID, ServiceLike] = {}

    def register_service(self, service: ServiceLike) -> None:
        self._services[service.id] = service

    def get_service(self, service_id: ServiceID) -> ServiceLike | None:
        return self._services.get(service_id)

    def remove_service(self, service_id: ServiceID) -> ServiceLike | None:
        return self._services.pop(service_id, None)

    def list_services(self) -> list[ServiceID]:
        return list(self._services.keys())

    def has_service(self, service_id: ServiceID) -> bool:
        return service_id in self._services

    def get_or_register_service(self, service_id: ServiceID, service_factory: Callable[[], ServiceLike]) -> ServiceLike:
        existing_service = self.get_service(service_id)
        if existing_service is not None:
            return existing_service

        service = service_factory()
        self.register_service(service)
        return service
