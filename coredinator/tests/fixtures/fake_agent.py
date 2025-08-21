#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys
import time


def _parse_args(argv: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {"config-name": None}
    it = iter(argv)
    for a in it:
        if a == "--config-name":
            try:
                out["config-name"] = next(it)
            except StopIteration:
                pass
    return out


def _install_sigterm_exit():
    def handler(signum: int, frame: object | None) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handler)


def main(argv: list[str]) -> int:
    _ = _parse_args(argv)

    mode = os.environ.get("FAKE_AGENT_BEHAVIOR", "long")
    if mode == "exit-0":
        return 0
    if mode == "exit-1":
        return 1

    _install_sigterm_exit()
    # Stay alive until killed; sleep in small increments to react to signals.
    try:
        while True:
            time.sleep(0.1)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
