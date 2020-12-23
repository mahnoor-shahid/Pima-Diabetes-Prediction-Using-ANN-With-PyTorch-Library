import torch
from torch.utils.data import Dataset
import numpy as np

class datasetLoader(Dataset):
    """
    loading and initializing dataset
    """
    def __init__(self, path):
        
        data = np.genfromtxt(path, delimiter=',', dtype=np.float32, filling_values=0.)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:,1:-1])
        self.y_data = torch.from_numpy(data[:,[-1]])
        print(np.unique(data[-1]))
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

