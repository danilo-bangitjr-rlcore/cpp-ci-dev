import logging
import time
from collections.abc import Callable, Iterable
from types import TracebackType

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress and outputs status via logging."""

    def __init__(
        self,
        total: int | None,
        desc: str = "",
        update_interval: int = 1,
        logger_instance: logging.Logger | None = None,
    ):
        """Initialize progress tracker with total iterations and configuration."""
        self.total = total
        self.desc = desc
        self.update_interval = update_interval
        self.logger = logger_instance or logger

        self.completed = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_completed = 0
        self.current_metrics: dict[str, float] = {}

    def update(self, n: int = 1, metrics: dict[str, float] | None = None):
        """Update progress by n iterations and optionally update metrics."""
        self.completed += n

        if metrics is not None:
            self.current_metrics.update(metrics)

        if (self.completed % self.update_interval == 0 or
            (self.total is not None and self.completed >= self.total)):
            self._log_progress()

    def _log_progress(self):
        """Log current progress status."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        elapsed_str = self._format_time(elapsed)

        if self.total is not None and self.completed > 0:
            # Calculate ETA based on average time per iteration
            avg_time_per_iter = elapsed / self.completed
            remaining_iters = self.total - self.completed
            eta = remaining_iters * avg_time_per_iter
            eta_str = self._format_time(eta)
        else:
            eta_str = "unknown"

        if self.total is None:
            total_str = 'unknown'
        else:
            total_str = str(self.total)

        prefix = f"{self.desc}: " if self.desc else ""
        progress_msg = (
            f"{prefix}{self.completed}/{total_str} "
            f"elapsed: {elapsed_str}, eta: {eta_str}"
        )

        # Add metrics if available
        if self.current_metrics:
            metrics_str = self._format_metrics(self.current_metrics)
            progress_msg += f", {metrics_str}"

        self.logger.info(progress_msg)

    @staticmethod
    def _format_metrics(metrics: dict[str, float]):
        """Format metrics dictionary for readable output."""
        formatted_pairs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if value >= 1000 or (0 < abs(value) < 0.01):
                    formatted_pairs.append(f"{key}: {value:.2e}")
                else:
                    formatted_pairs.append(f"{key}: {value:.3f}")
            else:
                formatted_pairs.append(f"{key}: {value}")
        return ", ".join(formatted_pairs)

    @staticmethod
    def _format_time(seconds: float):
        """Format time duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs:02d}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes:02d}m"

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit context manager and log final status."""
        if self.total is None or self.completed < self.total:
            # Log final progress if total is unknown or not at 100%
            self._log_progress()


def track(
    iterable: Iterable,
    desc: str = "",
    total: int | None = None,
    update_interval: int = 1,
    logger_instance: logging.Logger | None = None,
    *,
    metrics_callback: Callable[[object], dict[str, float]] | None = None,
) -> Iterable:
    """Track progress over an iterable with logging output."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            # Iterable doesn't support len(), convert to list
            iterable = list(iterable)
            total = len(iterable)

    with ProgressTracker(
        total=total,
        desc=desc,
        update_interval=update_interval,
        logger_instance=logger_instance,
    ) as tracker:
        for item in iterable:
            yield item
            metrics = metrics_callback(item) if metrics_callback else None
            tracker.update(metrics=metrics)
