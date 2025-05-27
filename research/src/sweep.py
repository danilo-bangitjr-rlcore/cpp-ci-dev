import argparse
import itertools
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from utils.dict import flatten

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True)
parser.add_argument("-o", "--overrides", type=str, required=True)
parser.add_argument("-c", "--command", type=str, default="python src/main.py")
parser.add_argument("-s", "--seeds", type=int, required=True)
parser.add_argument("-x", "--execute", action="store_true")
parser.add_argument("-q", "--quiet", action="store_true", help="Suppress subprocess output")
parser.add_argument("-m", "--max_workers", type=int, default=1)

args = parser.parse_args()


def run_command(command: str) -> int:
    """Run a command with optional output suppression."""
    if args.quiet:
        # Redirect stdout and stderr to devnull to suppress output
        return subprocess.run(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode
    else:
        # Run normally
        return os.system(command)


def generate_combinations(flattened_dict: dict[str, list[Any]]) -> list[list[tuple[str, Any]]]:
    keys = list(flattened_dict.keys())
    value_lists = [flattened_dict[key] for key in keys]

    # Generate all combinations of values
    value_combinations = list(itertools.product(*value_lists))

    # Convert to the desired output format
    result = []
    for values in value_combinations:
        combination = [(keys[i], values[i]) for i in range(len(keys))]
        result.append(combination)

    return result


def main():
    path = Path(args.overrides)
    with path.open("r") as f:
        override_cfg = yaml.safe_load(f)

    override_list = generate_combinations(flatten(override_cfg))
    commands = []

    for o in override_list:
        for seed in range(args.seeds):
            command = f"{args.command} -e {args.exp} -s {seed}"
            for k, v in o:
                command += f" {k}={v}"
            commands.append(command)

    # write commands to a file
    with open("commands.txt", "w") as f:
        for command in commands:
            f.write(command + "\n")

    # optionally execute the commands
    if args.execute:
        total_commands = len(commands)
        print(f"Executing {total_commands} commands with {args.max_workers} workers...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(run_command, cmd) for cmd in commands]

            # Process as they complete with progress bar
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="cmd"):
                pass


if __name__ == "__main__":
    main()
