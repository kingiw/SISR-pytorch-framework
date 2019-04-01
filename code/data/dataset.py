import os

import torch
import torch.utils.data as dataset

from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, path, transform=None):

        # List the file in the directory
        self.filelist = []
        for f in os.listdir(path):
            if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('jpg'):
                self.filelist.append(os.path.join(path, f))
        self.transform = transform
        
    def __getitem__(self, idx):
        # Say we are trying to load '/path/to/dataset/name.jpg'
        img_name = os.path.split(self.filelist[idx])[-1].split('.')[0]  # img_name = "name"
        
    def __len__(self):
        return len(self.filelist)
        