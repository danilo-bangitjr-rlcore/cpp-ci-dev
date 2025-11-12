# lib_process

`lib_process` provides a cleaner, more semantically meaningful wrapper around `psutil` for OS process manipulation.

## Purpose

This library wraps `psutil.Process` with a more intuitive API that:
- **Handles exceptions gracefully** - No more scattered try/except blocks for `NoSuchProcess` and `AccessDenied`
- **Returns sensible defaults** - Methods return `False` or empty lists instead of raising exceptions
- **Provides semantic clarity** - Method names and behavior clearly express intent
- **Simplifies common patterns** - Tree operations and termination waiting are first-class operations

## Core Features

### Process Creation
- `Process.start_in_background(args)` - Start a detached background process with stdin/stdout/stderr redirected to DEVNULL (cross-platform, handles Windows creationflags)

### Process Lifecycle
- `terminate()` - Send SIGTERM to process
- `kill()` - Send SIGKILL to process
- `terminate_tree()` - Terminate process and all children, wait for completion
- `kill_tree()` - Force kill process and all children, wait for completion
- `wait_for_termination()` - Wait for process to terminate with timeout and force-kill fallback

### Status Checks
- `is_running()` - Check if process is running (returns `False` on error)
- `is_zombie()` - Check if process is in zombie state (returns `False` on error)

### Process Tree
- `children()` - Get all descendants recursively (returns empty list on error)

### Properties
- `psutil` - Access underlying `psutil.Process` object when needed

### Utility Functions
- `find_processes_by_name_patterns(patterns)` - Find all processes matching any of the given name patterns (case-insensitive, from `lib_process.process_list`)

## Usage

```python
from lib_process.process import Process

# Start a background process (detached, no stdin/stdout/stderr)
process = Process.start_in_background(["python", "-m", "myapp", "--config", "config.yaml"])
print(f"Started process with PID: {process.psutil.pid}")

# Create from existing PID
process = Process.from_pid(12345)

# Check status without exception handling
if process.is_running():
    print("Process is alive")

# Terminate gracefully with automatic waiting
if process.terminate_tree(timeout=5.0):
    print("Process tree terminated successfully")
else:
    print("Timeout reached, force killed")

# Get all children without exception handling
for child in process.children():
    print(f"Child PID: {child.psutil.pid}")

# Find all processes matching patterns
from lib_process.process_list import find_processes_by_name_patterns

python_processes = find_processes_by_name_patterns(["python", "pytest"])
for proc in python_processes:
    print(f"Found: {proc.psutil.name()} (PID {proc.psutil.pid})")
```

## Design Philosophy

### Exception Handling
All methods handle `psutil.NoSuchProcess` and `psutil.AccessDenied` internally. Methods return sensible defaults instead of raising exceptions:
- Status checks return `False`
- Tree queries return empty lists
- Lifecycle operations silently succeed (process already dead is success)

### Semantic Clarity
- `terminate_tree()` clearly expresses "terminate this process and all children"
- `wait_for_termination()` with timeout and force-kill expresses the complete termination pattern
- `is_zombie()` is clearer than checking `status() == psutil.STATUS_ZOMBIE`

### Composability
Methods are designed to compose naturally:
```python
# Terminate all children, then parent
for child in process.children():
    child.terminate()
process.terminate()
```

This pattern is so common it's built-in as `terminate_tree()`.

## Comparison with psutil

### psutil (verbose)
```python
import psutil

try:
    proc = psutil.Process(pid)

    # Check if running
    try:
        if proc.is_running():
            print("Running")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print("Not running")

    # Get children
    try:
        for child in proc.children(recursive=True):
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

except psutil.NoSuchProcess:
    pass
```

### lib_process (clean)
```python
from lib_process.process import Process

process = Process.from_pid(pid)

if process.is_running():
    print("Running")

for child in process.children():
    child.terminate()
```

## Testing

All functionality is tested with real OS processes (no mocks). Tests spawn actual Python subprocesses and verify behavior end-to-end.
