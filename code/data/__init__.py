import os

import torch
from torch.utils import data

from PIL import Image
from torchvision import transforms

class Dataset(data.Dataset):
    def __init__(self, path, transform=None):

        # List the file in the directory
        self.filelist = []
        for f in os.listdir(path):
            if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('jpg'):
                self.filelist.append(os.path.join(path, f))
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        
    def __getitem__(self, idx):
        # Say we are trying to load '/path/to/dataset/name.jpg'
        img_name = os.path.split(self.filelist[idx])[-1].split('.')[0]  # img_name = "name"
        img = Image.open(self.filelist[idx])
        img = self.transform(img) # IMG [0, 255] -> Tensor [0,1]
        
        return img_name, img

    def __len__(self):
        return len(self.filelist)

class LRHR_Dataset(data.Dataset):
    """
    LR image and HR image should have the same filename
    """
    def __init__(self, LR_path, HR_path, transform=None):
        super(LRHR_Dataset, self).__init__()
        self.LR_set = Dataset(LR_path, transform)
        self.HR_set = Dataset(HR_path, transform)
        assert(len(self.LR_set) == len(self.HR_set))

    def __getitem__(self, idx):
        img_name, HR = self.HR_set[idx]
        _, LR = self.LR_set[idx]
        return img_name, LR, HR
    
    def __len__(self):
        return len(self.LR_set)


