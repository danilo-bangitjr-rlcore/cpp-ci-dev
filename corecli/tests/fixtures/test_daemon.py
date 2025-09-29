#!/usr/bin/env python3
"""
Simple test daemon process for testing daemon utilities.
Supports different behaviors controlled by environment variables.
"""

import os
import signal
import sys
import time
from types import FrameType


def install_signal_handlers():
    """Install signal handlers based on behavior mode."""
    behavior = os.environ.get("TEST_DAEMON_BEHAVIOR", "normal")

    if behavior == "ignore_sigterm":
        # Ignore SIGTERM but respond to SIGKILL
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif behavior == "normal":
        # Normal graceful shutdown on SIGTERM
        def handler(signum: int, frame: FrameType | None) -> None:
            sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
    # For "no_signals", don't install any handlers


def main():
    """Main entry point for test daemon."""
    behavior = os.environ.get("TEST_DAEMON_BEHAVIOR", "normal")

    # Write PID file if requested
    pid_file = os.environ.get("TEST_DAEMON_PID_FILE")
    if pid_file:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

    # Install appropriate signal handlers
    install_signal_handlers()

    # Handle different exit behaviors
    if behavior == "exit_immediately":
        sys.exit(0)
    elif behavior == "exit_with_error":
        sys.exit(1)

    # For long-running behaviors, sleep in small intervals
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
