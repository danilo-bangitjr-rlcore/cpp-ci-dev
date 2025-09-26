import subprocess
from pathlib import Path


class CliRunner:
    def __init__(self, monorepo_root: Path):
        self.monorepo_root = monorepo_root
        self.corecli_path = monorepo_root / "corecli"
        self.cli_binary = self.corecli_path / ".venv" / "bin" / "corecli"

    def run_command(
        self,
        *args: str,
        cwd: Path | None = None,
        check: bool = True,
    ):
        """
        Run a corecli command using the installed CLI binary.
        """
        if cwd is None:
            cwd = self.monorepo_root

        cmd = [str(self.cli_binary), *args]
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)

    def start_coredinator(self, port: int, log_file: Path, base_path: Path, check: bool = True):
        args = ["coredinator", "start", "--port", str(port), "--log-file", str(log_file), "--base-path", str(base_path)]
        return self.run_command(*args, check=check)

    def stop_coredinator(self, port: int, check: bool = False):
        return self.run_command("coredinator", "stop", "--port", str(port), check=check)

    def status_coredinator(self, port: int, check: bool = True):
        return self.run_command("coredinator", "status", "--port", str(port), check=check)
