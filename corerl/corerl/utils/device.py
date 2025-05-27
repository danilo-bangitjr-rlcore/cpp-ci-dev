import logging

import torch

log = logging.getLogger(__name__)


class Device:
    def __init__(self):
        self.device = torch.device('cpu')

    def update_device(self, device_name: str | None = None):
        if device_name:
            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        log.info(f"Update Device: {self.device}")

device = Device()
