# lib_progress

Fast, lightweight progress tracking using logging instead of progress bars.

A simple replacement for tqdm that outputs progress information through the logger instead of updating progress bars. Shows iterations completed/total, elapsed time, and ETA based on simple averaging.

## Usage

```python
from lib_progress import ProgressTracker, track

# Using context manager
with ProgressTracker(total=100, desc="Processing", update_interval=10) as tracker:
    for i in range(100):
        # Do work here
        tracker.update()

# Using convenience function
for item in track(items, desc="Processing items", update_interval=5):
    # Process item
    pass
```
