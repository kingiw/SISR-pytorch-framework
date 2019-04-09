import torch
from torch.utils.data import DataLoader

from data import Dataset
from options import args

# from trainer import Trainer

torch.manual_seed(args.seed)
train_set = Dataset(args.train_data_path)
train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)


# for filenames, imgs in dataloader:
#     pass
