from subprocess import Popen
from typing import NewType

AgentID = NewType("AgentID", str)

class AgentProcess:
    def __init__(self, id: AgentID):
        self.id = id

        self._process: Popen | None = None


    def start(self):
        ...

    def stop(self):
        ...

    def restart(self):
        ...

    def status(self):
        ...
