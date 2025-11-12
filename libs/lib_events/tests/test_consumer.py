import threading
import time
from queue import Queue
from typing import Any
from unittest.mock import Mock

import zmq

from lib_events.legacy.consumer_task import consumer_task


def test_consumer_task_receives_message(mocker: Mock):
    # Setup mocks
    topic = "testtopic"
    message = b"testtopic hello"
    event = mocker.MagicMock()

    # Mock socket
    mock_socket = mocker.MagicMock()
    mock_socket.poll.return_value = 1  # Message available
    mock_socket.recv.return_value = message

    # Mock event class
    mock_event_class = mocker.MagicMock()
    mock_event_class.model_validate_json.return_value = event

    # Setup real queue and event
    queue = Queue()
    stop_event = threading.Event()

    # Run consumer in a thread
    thread = threading.Thread(
        target=consumer_task,
        args=(mock_socket, queue, stop_event),
        kwargs={
            'event_class': mock_event_class,
            'topic': topic,
        },
        daemon=True,
    )
    thread.start()

    # Allow time for processing
    time.sleep(0.2)

    # Cleanup
    stop_event.set()
    thread.join(timeout=1)

    # Assert
    mock_socket.setsockopt_string.assert_called_once()
    assert mock_socket.setsockopt_string.call_args[0][1] == topic
    mock_event_class.model_validate_json.assert_called_with(message.split(b" ", 1)[1])
    mock_socket.recv.assert_called()
    assert not queue.empty()
    assert queue.get(block=False) == event

    queue.shutdown()

def test_consumer_task_no_message(mocker: Mock):
    # Setup mocks
    topic = "testtopic"

    # Mock socket that returns no message
    mock_socket = mocker.MagicMock()
    mock_socket.poll.return_value = 0

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Setup real queue and event
    queue = Queue()
    stop_event = threading.Event()

    # Run consumer in a thread
    thread = threading.Thread(
        target=consumer_task,
        args=(mock_socket, queue, stop_event),
        kwargs={
            'event_class': mock_event_class,
            'topic': topic,
        },
        daemon=True,
    )
    thread.start()

    # Allow time for processing
    time.sleep(0.2)

    # Cleanup
    stop_event.set()
    thread.join(timeout=1)

    # Assert
    mock_socket.setsockopt_string.assert_called_once()
    assert mock_socket.setsockopt_string.call_args[0][1] == topic
    mock_socket.recv.assert_not_called()  # Since poll returns 0
    mock_event_class.model_validate_json.assert_not_called()
    assert queue.empty()

    queue.shutdown()

def test_consumer_task_handles_error(mocker: Mock):
    # Setup mocks
    topic = "testtopic"

    # Mock socket that raises ZMQError
    mock_socket = mocker.MagicMock()
    mock_socket.poll.return_value = 1

    # Create a ZMQError for testing
    zmq_error = Mock(side_effect=zmq.error.ZMQError(zmq.EINVAL))
    mock_socket.recv.side_effect = zmq_error

    # Mock event class
    mock_event_class = mocker.MagicMock()

    # Setup a consumer task that captures errors
    error = []
    def error_capturing_consumer(*args: Any, **kwargs: Any) -> None:
        try:
            consumer_task(*args, **kwargs)
        except Exception as e:
            error.append(e)

    # Setup real queue and event
    queue = Queue()
    stop_event = threading.Event()

    # Run consumer in a thread
    thread = threading.Thread(
        target=error_capturing_consumer,
        args=(mock_socket, queue, stop_event),
        kwargs={
            'event_class': mock_event_class,
            'topic': topic,
        },
        daemon=True,
    )
    thread.start()

    # Allow time for processing
    time.sleep(0.2)

    # Cleanup
    stop_event.set()
    thread.join(timeout=1)

    # Assert
    assert queue.empty()
    assert len(error) == 1
    assert isinstance(error[0], zmq.error.ZMQError)

    queue.shutdown()
