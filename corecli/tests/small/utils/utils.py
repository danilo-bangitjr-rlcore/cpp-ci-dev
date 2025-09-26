import os
import subprocess
import sys

# Test utilities

def start_test_daemon(behavior: str = "normal") -> subprocess.Popen[bytes]:
    script_path = os.path.join(os.path.dirname(__file__), "../fixtures/test_daemon.py")
    env = dict(os.environ)
    env["TEST_DAEMON_BEHAVIOR"] = behavior
    return subprocess.Popen([sys.executable, script_path], env=env)


def cleanup_process(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=1)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass
