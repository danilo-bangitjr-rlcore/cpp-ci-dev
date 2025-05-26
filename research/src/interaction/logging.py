import csv
import time
from datetime import datetime
from pathlib import Path


def setup_logging() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    csv_path = log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'measurement', 'tags', 'fields'])

    return csv_path

def log_to_file(log_path: Path, measurement: str, tags: dict, fields: dict):
    timestamp = int(time.time() * 1000)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            measurement,
            str(tags),
            str(fields)
        ])
