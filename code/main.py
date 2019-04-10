import torch
from torch.utils.data import DataLoader

from data import LRHR_Dataset
from options import args
import utils
import os
import logging
from model import create_model
from loss import Loss
from trainer import Trainer

# from trainer import Trainer

torch.manual_seed(args.seed)

# Set up the training / validation dataloader
train_set = LRHR_Dataset(args.train_LR, args.train_HR)
train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
# val_set = LRHR_Dataset(args.val_LR, args.val_HR)
# val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)

# Logger
if not os.path.isdir('../experiments/' + args.name):
    os.mkdir('../experiments/' + args.name)
utils.setup_logger('base', '../experiments/' + args.name, utils.get_timestamp(), screen=True)
logger = logging.getLogger('base')

# Model
model = create_model(args)

loss = loss.Loss(args)

trainer = Tranier(args, train_dataloader, val_dataloader, model, loss)


# for filenames, LR, HR in train_dataloader:
#     print(filenames)
#     print(LR[0][0][:1][:1])
#     print(HR[0][0][:1][:1])