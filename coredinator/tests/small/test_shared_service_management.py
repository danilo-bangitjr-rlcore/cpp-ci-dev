from pathlib import Path

from coredinator.service.protocols import ServiceBundleID, ServiceID
from coredinator.service.service_manager import ServiceManager


class MockServiceBundle:
    """Lightweight ServiceBundle implementation for testing ownership logic."""

    def __init__(self, bundle_id: ServiceBundleID, required_services: set[ServiceID]):
        self.id = bundle_id
        self._required_services = required_services

    def get_required_services(self) -> set[ServiceID]:
        return self._required_services


class TestServiceManagerOwnership:
    def test_service_manager_tracks_ownership(self, tmp_path: Path):
        """
        Test that ServiceManager properly tracks service ownership.
        """
        service_manager = ServiceManager(tmp_path)

        # Create a test bundle
        bundle = MockServiceBundle(
            ServiceBundleID("test-bundle"),
            {ServiceID("service-1"), ServiceID("service-2")},
        )

        # Register the bundle
        service_manager.register_bundle(bundle)

        # Check ownership tracking
        assert service_manager.get_service_owners(ServiceID("service-1")) == {ServiceBundleID("test-bundle")}
        assert service_manager.get_service_owners(ServiceID("service-2")) == {ServiceBundleID("test-bundle")}
        assert service_manager.get_bundle_services(ServiceBundleID("test-bundle")) == {
            ServiceID("service-1"),
            ServiceID("service-2"),
        }

    def test_multiple_bundles_sharing_service(self, tmp_path: Path):
        """
        Test that multiple bundles can share the same service.
        """
        service_manager = ServiceManager(tmp_path)
        # Create two test bundles that share a service
        bundle1 = MockServiceBundle(
            ServiceBundleID("bundle-1"),
            {ServiceID("shared-service")},
        )
        bundle2 = MockServiceBundle(
            ServiceBundleID("bundle-2"),
            {ServiceID("shared-service")},
        )

        # Register both bundles
        service_manager.register_bundle(bundle1)
        service_manager.register_bundle(bundle2)

        # Check that service is shared
        assert service_manager.is_service_shared(ServiceID("shared-service"))
        assert service_manager.get_service_owners(ServiceID("shared-service")) == {
            ServiceBundleID("bundle-1"),
            ServiceBundleID("bundle-2"),
        }
        assert not service_manager.can_stop_service(ServiceID("shared-service"))
