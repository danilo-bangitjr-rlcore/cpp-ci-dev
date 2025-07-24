
from enum import auto
from queue import Queue
from unittest.mock import Mock

from lib_defs.type_defs.base_events import BaseEventTopic

from lib_utils.messages.base_event_bus import BaseEventBus


class SampleTopics(BaseEventTopic):
        test_topic = auto()

def test_base_event_bus_init(mocker: Mock):
    # Mock ZMQ components
    mock_context = mocker.MagicMock()
    mock_sub_socket = mocker.MagicMock()
    mock_pub_socket = mocker.MagicMock()
    mock_context.socket.side_effect = [mock_sub_socket, mock_pub_socket]

    mocker.patch('zmq.Context', return_value=mock_context)
    mock_thread_class = mocker.patch('threading.Thread')

    # Mock event class
    mock_event_class = mocker.MagicMock()


    # Create event bus
    event_bus = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
        consumer_name="test_consumer",
        subscriber_sockets=["tcp://*:5555", "tcp://*:5556"],
    )

    # Assert initialization
    assert isinstance(event_bus.queue, Queue)
    assert event_bus.zmq_context == mock_context
    assert event_bus.subscriber_socket == mock_sub_socket
    assert event_bus.publisher_socket == mock_pub_socket

    mock_thread_class.assert_called_once()
    call_kwargs = mock_thread_class.call_args.kwargs

    assert call_kwargs['daemon'] is True
    assert call_kwargs['name'] == "test_consumer"

    # Assert socket bindings
    assert mock_sub_socket.bind.call_count == 2
    mock_sub_socket.bind.assert_any_call("tcp://*:5555")
    mock_sub_socket.bind.assert_any_call("tcp://*:5556")


def test_base_event_bus_init_no_subscriber_sockets(mocker: Mock):
    # Mock ZMQ components
    mock_context = mocker.MagicMock()
    mock_sub_socket = mocker.MagicMock()
    mock_pub_socket = mocker.MagicMock()
    mock_context.socket.side_effect = [mock_sub_socket, mock_pub_socket]

    mocker.patch('zmq.Context', return_value=mock_context)
    mocker.patch('threading.Thread')

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Create event bus without subscriber sockets
    _ = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
    )

    # Assert no socket bindings
    mock_sub_socket.bind.assert_not_called()


def test_base_event_bus_start(mocker: Mock):
    # Mock dependencies
    mocker.patch('zmq.Context')
    mock_thread = mocker.MagicMock()
    mocker.patch('threading.Thread', return_value=mock_thread)

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Create and start event bus
    event_bus = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
    )
    event_bus.start()

    # Assert thread started
    mock_thread.start.assert_called_once()


def test_recv_event_with_event(mocker: Mock):
    # Mock dependencies
    mocker.patch('zmq.Context')
    mocker.patch('threading.Thread')

    # Mock event class and event
    mock_event_class = mocker.MagicMock()
    mock_event = mocker.MagicMock()

    # Create event bus
    event_bus = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
    )

    # Put event in queue
    event_bus.queue.put(mock_event)

    # Receive event
    result = event_bus.recv_event()

    # Assert
    assert result == mock_event


def test_recv_event_empty_queue(mocker: Mock):
    # Mock dependencies
    mocker.patch('zmq.Context')
    mocker.patch('threading.Thread')

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Create event bus
    event_bus = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
    )

    # Receive from empty queue
    result = event_bus.recv_event()

    # Assert
    assert result is None


def test_recv_event_stop_event_set(mocker: Mock):
    # Mock dependencies
    mocker.patch('zmq.Context')
    mocker.patch('threading.Thread')

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Create event bus
    event_bus = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
    )

    # Set stop event
    event_bus.stop_event.set()

    # Receive event
    result = event_bus.recv_event()

    # Assert
    assert result is None


def test_cleanup(mocker: Mock):
    # Mock dependencies
    mock_context = mocker.MagicMock()
    mock_sub_socket = mocker.MagicMock()
    mock_pub_socket = mocker.MagicMock()
    mock_context.socket.side_effect = [mock_sub_socket, mock_pub_socket]
    mock_thread = mocker.MagicMock()

    mocker.patch('zmq.Context', return_value=mock_context)
    mocker.patch('threading.Thread', return_value=mock_thread)

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Create event bus
    event_bus = BaseEventBus(
        event_class=mock_event_class,
        topic=SampleTopics.test_topic,
    )

    # Put some events in queue
    event_bus.queue.put("event1")
    event_bus.queue.put("event2")

    # Cleanup
    event_bus.cleanup()

    # Assert cleanup actions
    assert event_bus.stop_event.is_set()
    assert event_bus.queue.empty()
    mock_thread.join.assert_called_once_with(timeout=5)
