import torch
from torch.utils.data import DataLoader
from data import Dataset
from options import args

import os
from model import create_model
import utils


# for k, v in vars(args).items():
#     print("{}: {}".format(k, v))

src = args.test_src
dest = os.path.join(args.test_dest, args.name)
if not os.path.isdir(dest):
    os.mkdir(dest)

dataset = Dataset(src)
loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

model = create_model(args)
model.eval()
if args.pre_train_model != "...":
    print("Loading pretrained model ... ")
    model.load_state_dict(torch.load(args.pre_train_model))

print("Testing...")
with torch.no_grad():
    for i, (name, lr) in enumerate(loader):
        if not args.cpu and args.n_GPUs > 0:
            lr = lr.cuda()
        sr = model(lr)
        
        utils.save_image(sr, dest, name)
        if (i+1) % 10 == 0:
            print("Test Progess [{}/{}]".format((i+1) * args.test_batch_size, len(dataset)))

        if (i+1) * args.test_batch_size > len(dataset):
            print("Done")
            break
    