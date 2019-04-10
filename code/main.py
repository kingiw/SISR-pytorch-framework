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


torch.manual_seed(args.seed)

# Init
for s in ['', 'model', 'optimizer', 'tb_logger', 'log', 'results']:
    utils.make_directory('../experiments/{}/{}'.format(args.name,s))


# Set up the training / validation dataloader
train_set = LRHR_Dataset(args.train_LR, args.train_HR)
train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_set = LRHR_Dataset(args.val_LR, args.val_HR)
val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

# Logger
if not os.path.isdir('../experiments/' + args.name):
    os.mkdir('../experiments/' + args.name)
utils.setup_logger('base', '../experiments/{}/log'.format(args.name), utils.get_timestamp(), screen=True)
logger = logging.getLogger('base')

logger.info(args)

my_model = create_model(args)
my_loss = Loss(args)
trainer = Trainer(args, train_dataloader, val_dataloader, my_model, my_loss)
trainer.train()