import numpy as np
import os
from torch.utils.data import Dataset

class GFVDataset(Dataset):
    def __init__(self, config):
        self.dataset_path = config["dataset"]["path"]["train"]["image"]

        self.dataset = []
        for file in os.listdir(self.dataset_path):
            if file.endswith(".npy"):
                data_path = os.path.join(self.dataset_path, file)
                self.dataset.append(data_path)

    def __getitem__(self, index):
        data_path = self.dataset[index]
        data = np.load(data_path)
        return data

    def __len__(self):
        return len(self.dataset)