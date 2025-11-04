"""Tests for RLTuneService base class."""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, Mock, patch

import pytest
from lib_defs.type_defs.base_events import EventTopic

from service_framework.service import RLTuneService, ServiceState


# Concrete implementation for testing
class MockService(RLTuneService):
    def __init__(
        self,
        service_name: str = "test_service",
        event_bus_host: str = "localhost",
        should_fail_start: bool = False,
        should_fail_stop: bool = False,
    ):
        super().__init__(
            service_name=service_name,
            event_topic=EventTopic.corerl,
            event_bus_host=event_bus_host,
        )
        self.should_fail_start = should_fail_start
        self.should_fail_stop = should_fail_stop
        self.start_called = False
        self.stop_called = False
        self.run_called = False

    async def _do_start(self) -> None:
        self.start_called = True
        if self.should_fail_start:
            raise RuntimeError("Intentional start failure")

    async def _do_stop(self) -> None:
        self.stop_called = True
        if self.should_fail_stop:
            raise RuntimeError("Intentional stop failure")

    async def _do_run(self) -> AsyncGenerator[None]:
        self.run_called = True
        yield


@pytest.mark.asyncio
class TestRLTuneServiceLifecycle:
    """Test service lifecycle methods."""

    @pytest.mark.timeout(5)
    async def test_initial_state(self):
        service = MockService()
        assert service.get_state() == ServiceState.CREATED
        assert not service.is_running()
        assert not service.is_stopped()
        assert not service.is_failed()
        assert service.get_error() is None

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_successful_start(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService()
        await service.start()

        assert service.start_called
        assert service.run_called
        assert service.get_state() == ServiceState.RUNNING
        assert service.is_running()
        mock_client.connect.assert_called_once()

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_successful_stop(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService()
        await service.start()
        await service.stop()

        assert service.stop_called
        assert service.get_state() == ServiceState.STOPPED
        assert service.is_stopped()
        assert not service.is_running()
        mock_client.close.assert_called_once()

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_start_failure(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService(should_fail_start=True)

        with pytest.raises(RuntimeError, match="Intentional start failure"):
            await service.start()

        assert service.start_called
        assert service.get_state() == ServiceState.FAILED
        assert service.is_failed()
        assert service.get_error() is not None

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_stop_failure(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService(should_fail_stop=True)
        await service.start()

        with pytest.raises(RuntimeError, match="Intentional stop failure"):
            await service.stop()

        assert service.stop_called
        assert service.get_state() == ServiceState.FAILED
        assert service.is_failed()

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_cannot_start_twice(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService()
        await service.start()

        with pytest.raises(RuntimeError, match="Cannot start service"):
            await service.start()

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_stop_idempotent(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService()
        await service.start()
        await service.stop()
        await service.stop()

        assert service.get_state() == ServiceState.STOPPED


@pytest.mark.asyncio
class TestRLTuneServiceStateQueries:
    """Test state query methods."""

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_state_transitions(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService()

        # CREATED
        assert service.get_state() == ServiceState.CREATED
        assert not service.is_running()
        assert not service.is_stopped()
        assert not service.is_failed()

        # RUNNING
        await service.start()
        assert service.get_state() == ServiceState.RUNNING
        assert service.is_running()
        assert not service.is_stopped()
        assert not service.is_failed()

        # STOPPED
        await service.stop()
        assert service.get_state() == ServiceState.STOPPED
        assert not service.is_running()
        assert service.is_stopped()
        assert not service.is_failed()

    @pytest.mark.timeout(5)
    @patch('service_framework.service.EventBusClient')
    async def test_failed_state_queries(self, mock_event_bus_client: Mock):
        mock_client = MagicMock()
        mock_event_bus_client.return_value = mock_client

        service = MockService(should_fail_start=True)

        with pytest.raises(RuntimeError):
            await service.start()

        assert service.get_state() == ServiceState.FAILED
        assert not service.is_running()
        assert not service.is_stopped()
        assert service.is_failed()
        assert service.get_error() is not None
        assert "Intentional start failure" in str(service.get_error())
