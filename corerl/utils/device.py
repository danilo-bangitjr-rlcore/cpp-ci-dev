import torch

class Device:
    def __init__(self):
        self.device = torch.device('cpu')

    def update_device(self, device_name=None):
        print("device_name:", device_name)
        if device_name:
            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("update_device device:")
        print(self.device)

device = Device()