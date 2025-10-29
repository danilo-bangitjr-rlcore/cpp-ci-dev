import sys
import time

import servicemanager
import uvicorn
import win32service
import win32serviceutil
from server.core_ui import get_app


class MyService:
    def __init__(self):
        self.running = False
        self.app = get_app()

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        uvicorn.run(self.app, host="127.0.0.1", port=8000)
        while self.running:
            time.sleep(10)
            servicemanager.LogInfoMsg("Service running...")
        servicemanager.LogInfoMsg("Service stopping...")

class MyServiceFramework(win32serviceutil.ServiceFramework):
    _svc_name_ = 'CoreUI6'
    _svc_display_name_ = 'CoreUI6'

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.service_impl.stop()
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def SvcDoRun(self):
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        self.service_impl = MyService()
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        self.service_impl.run()

def init():
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MyServiceFramework)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(MyServiceFramework)

if __name__ == '__main__':
    init()
