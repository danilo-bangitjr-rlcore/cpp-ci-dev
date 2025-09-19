"""Example usage of lib_progress."""

import logging
import random
import time

from lib_progress import ProgressTracker, track

# Configure logging to see the output
logging.basicConfig(level=logging.INFO, format='%(message)s')


def example_context_manager():
    """Example using context manager."""
    print("\n=== Context Manager Example ===")

    with ProgressTracker(total=10, desc="Processing items", update_interval=3) as tracker:
        for _ in range(10):
            # Simulate some work
            time.sleep(0.1)
            tracker.update()


def example_track_function():
    """Example using track function."""
    print("\n=== Track Function Example ===")

    items = list(range(8))
    for _ in track(items, desc="Processing list", update_interval=2):
        # Simulate some work
        time.sleep(0.1)


def example_generator():
    """Example with generator that doesn't support len()."""
    print("\n=== Generator Example ===")

    def data_generator():
        for i in range(6):
            yield f"item_{i}"
            time.sleep(0.1)

    for _ in track(data_generator(), desc="Processing generator"):
        # Process each item
        pass


# ---------------------------------------------------------------------------- #
#                            Metric Logging Examples                           #
# ---------------------------------------------------------------------------- #


def example_context_manager_with_metrics():
    """Example using context manager with metrics."""
    print("\n=== Context Manager with Metrics Example ===")

    with ProgressTracker(total=8, desc="Training model", update_interval=2) as tracker:
        for epoch in range(8):
            # Simulate training with varying metrics
            loss = 1.0 - (epoch * 0.1) + random.uniform(-0.05, 0.05)
            accuracy = 0.5 + (epoch * 0.06) + random.uniform(-0.02, 0.02)
            learning_rate = 0.001 * (0.9 ** epoch)

            time.sleep(0.1)  # Simulate work

            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "lr": learning_rate,
            }
            tracker.update(metrics=metrics)


def example_track_with_metrics():
    """Example using track function with metrics callback."""
    print("\n=== Track Function with Metrics Callback Example ===")

    # Simulate processing a dataset
    dataset = [{"data": f"sample_{i}", "quality": random.uniform(0.7, 1.0)} for i in range(6)]

    def extract_metrics(item):
        """Extract metrics from each processed item."""
        return {
            "quality": item["quality"],
            "batch_size": 32,
        }

    for _ in track(
        dataset,
        desc="Processing dataset",
        update_interval=2,
        metrics_callback=extract_metrics,
    ):
        # Simulate processing
        time.sleep(0.1)


def example_mixed_metrics():
    """Example showing different metric formats."""
    print("\n=== Mixed Metrics Format Example ===")

    with ProgressTracker(total=5, desc="Analysis", update_interval=1) as tracker:
        for i in range(5):
            # Different scales of metrics to show formatting
            metrics = {
                "large_val": 1234567.89,  # Should use scientific notation
                "small_val": 0.000123,    # Should use scientific notation
                "normal_val": 0.876,      # Should use normal formatting
                "score": i * 0.2,         # Regular incrementing value
            }

            time.sleep(0.1)
            tracker.update(metrics=metrics)


if __name__ == "__main__":
    example_context_manager()
    example_track_function()
    example_generator()
    example_context_manager_with_metrics()
    example_track_with_metrics()
    example_mixed_metrics()
