import logging
import time
from unittest.mock import Mock

from lib_progress.tracker import ProgressTracker, format_time, track


def test_progress_tracker_basic_functionality():
    """Test basic progress tracking functionality."""
    mock_logger = Mock(spec=logging.Logger)

    tracker = ProgressTracker(
        total=10,
        desc="Test",
        update_interval=1,
        logger_instance=mock_logger,
    )

    # Test initial state
    assert tracker.completed == 0
    assert tracker.total == 10
    assert tracker.desc == "Test"

    # Test update
    tracker.update()
    assert tracker.completed == 1

    # Should have logged progress
    mock_logger.info.assert_called()
    call_args = mock_logger.info.call_args[0][0]
    assert "Test: 1/10" in call_args
    assert "elapsed:" in call_args
    assert "eta:" in call_args


def test_progress_tracker_with_update_interval():
    """Test progress tracking with custom update interval."""
    mock_logger = Mock(spec=logging.Logger)

    tracker = ProgressTracker(
        total=10,
        update_interval=3,
        logger_instance=mock_logger,
    )

    # First two updates shouldn't log
    tracker.update()
    tracker.update()
    mock_logger.info.assert_not_called()

    # Third update should log
    tracker.update()
    mock_logger.info.assert_called_once()


def test_progress_tracker_context_manager():
    """Test progress tracker as context manager."""
    mock_logger = Mock(spec=logging.Logger)

    with ProgressTracker(
        total=5,
        desc="Context test",
        update_interval=1,
        logger_instance=mock_logger,
    ) as tracker:
        tracker.update(2)
        assert tracker.completed == 2

    # Should have logged at least once during updates
    assert mock_logger.info.call_count >= 1


def test_track_function():
    """Test the track convenience function."""
    mock_logger = Mock(spec=logging.Logger)

    items = [1, 2, 3, 4, 5]
    result = list(track(
        items,
        desc="Tracking items",
        update_interval=2,
        logger_instance=mock_logger,
    ))

    assert result == items
    # Should have logged at least once (items 2, 4, and final)
    assert mock_logger.info.call_count >= 1


def test_format_time():
    """Test time formatting utility."""
    # Test seconds
    assert format_time(5.5) == "5.5s"
    assert format_time(45.2) == "45.2s"

    # Test minutes
    assert format_time(65) == "1m05s"
    assert format_time(125) == "2m05s"

    # Test hours
    assert format_time(3665) == "1h01m"
    assert format_time(7325) == "2h02m"


def test_track_with_no_total():
    """Test track function when total is not provided."""
    mock_logger = Mock(spec=logging.Logger)

    # List that supports len()
    items = [1, 2, 3]
    result = list(track(items, logger_instance=mock_logger))
    assert result == items

    # Tuple also supports len()
    items_tuple = (1, 2, 3)
    result = list(track(items_tuple, logger_instance=mock_logger))
    assert result == [1, 2, 3]


def test_track_with_generator():
    """Test track function with generator (unknown total)."""
    mock_logger = Mock(spec=logging.Logger)

    # Generator that doesn't support len()
    def gen():
        yield from [1, 2, 3]

    result = list(track(gen(), logger_instance=mock_logger))
    assert result == [1, 2, 3]

    # Should have logged progress (without total/ETA)
    assert mock_logger.info.call_count >= 1


def test_eta_calculation():
    """Test ETA calculation accuracy."""
    mock_logger = Mock(spec=logging.Logger)

    tracker = ProgressTracker(
        total=4,
        update_interval=1,
        logger_instance=mock_logger,
    )

    # Simulate some time passing between updates
    start_time = time.time()
    tracker.start_time = start_time - 1.0  # Pretend 1 second has passed

    tracker.update(2)  # 2 out of 4 completed

    # Should have calculated ETA
    mock_logger.info.assert_called()
    call_args = mock_logger.info.call_args[0][0]
    assert "eta:" in call_args


def test_progress_tracker_with_metrics():
    """Test progress tracking with metrics."""
    mock_logger = Mock(spec=logging.Logger)

    tracker = ProgressTracker(
        total=5,
        desc="Test with metrics",
        update_interval=1,
        logger_instance=mock_logger,
    )

    # Update with metrics
    metrics = {"loss": 0.5, "accuracy": 0.85, "lr": 0.001}
    tracker.update(metrics=metrics)

    # Should have logged progress with metrics
    mock_logger.info.assert_called()
    call_args = mock_logger.info.call_args[0][0]
    assert "loss: 0.500" in call_args
    assert "accuracy: 0.850" in call_args
    assert "lr: 1.00e-03" in call_args


def test_progress_tracker_unknown_total():
    """Test progress tracking with unknown total."""
    mock_logger = Mock(spec=logging.Logger)

    tracker = ProgressTracker(
        total=None,
        desc="Unknown total test",
        update_interval=2,
        logger_instance=mock_logger,
    )

    # Update a few times
    tracker.update()
    mock_logger.info.assert_not_called()  # Not at interval yet

    tracker.update()
    mock_logger.info.assert_called()


def test_format_metrics():
    """Test metrics formatting."""
    # Test normal values
    metrics = {"score": 0.876, "count": 42.0}
    formatted = ProgressTracker._format_metrics(metrics)
    assert "score: 0.876" in formatted
    assert "count: 42.000" in formatted

    # Test scientific notation for large values
    metrics = {"big_val": 1234567.89}
    formatted = ProgressTracker._format_metrics(metrics)
    assert "1.23e+06" in formatted

    # Test scientific notation for small values
    metrics = {"small_val": 0.000123}
    formatted = ProgressTracker._format_metrics(metrics)
    assert "1.23e-04" in formatted


def test_track_with_metrics_callback():
    """Test track function with metrics callback."""
    mock_logger = Mock(spec=logging.Logger)

    items = [{"quality": 0.9}, {"quality": 0.8}, {"quality": 0.95}]

    def extract_metrics(item: dict[str, float]):
        return {"quality": item["quality"]}

    result = list(track(
        items,
        desc="Test tracking",
        update_interval=1,
        logger_instance=mock_logger,
        metrics_callback=extract_metrics,
    ))

    assert result == items
    # Should have logged progress with metrics
    assert mock_logger.info.call_count >= 1

    # Check that metrics were included in at least one log call
    logged_any_metrics = any(
        "quality:" in call.args[0]
        for call in mock_logger.info.call_args_list
    )
    assert logged_any_metrics
