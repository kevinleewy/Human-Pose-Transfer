import numpy as np
import os
from torch.utils.data import Dataset

class GFVDataset(Dataset):
    def __init__(self, config):
        self.data_path = config["dataset"]["path"]["train"]["image"]
        self.imsize = config["model"]["lgan"]["imsize"]

        self.dataset = []
        for file in os.listdir(self.data_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(self.data_path, file))
                self.dataset.append(data)

    def __getitem__(self, index):
        data = self.dataset[index]
        # data = data.reshape(-1, self.imsize, self.imsize)
        return data

    def __len__(self):
        return len(self.dataset)