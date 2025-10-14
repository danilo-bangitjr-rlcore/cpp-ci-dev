import psutil
import platform

def display_system_stats():
    print(f"System: {platform.system()}\n")
    
    # CPU utilization
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # RAM utilization
    ram = psutil.virtual_memory()
    print(f"RAM Usage: {ram.percent}% ({ram.used / (1024**3):.2f}GB / {ram.total / (1024**3):.2f}GB)")
    
    # Disk utilization
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}% ({disk.used / (1024**3):.2f}GB / {disk.total / (1024**3):.2f}GB)")

if __name__ == "__main__":
    display_system_stats()
