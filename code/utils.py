import os
import torchvision
import matplotlib.pyplot as plt 
import numpy as np
import logging
import os
from datetime import datetime
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs



class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret
        
    def reset(self):
        self.acc = 0

def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_timestamp():
    '''
    Return the current time in format "year-month-day-hour-minute-second"
    '''
    return datetime.now().strftime('%y-%m-%d-%H-%M-%S')

def setup_logger(logger_name, root, file_name, level=logging.INFO, screen=False):
    '''set up logger'''
    if not os.path.exists(root):
        os.mkdir(root)
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, file_name + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    else:
        NotImplementedError('Optimizer [{:s}] not recognized.'.format(args.optimizer))

    kwargs['lr'] = args.lr
    kwargs['weight_decay']  = args.weight_decay

    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size = args.lr_decay,
            gamma = args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones = milestones,
            gamma = args.gamma
        )
    return scheduler


def save_image(img_tensor_list, path, filename_list, img_type='jpg'):
    """
    img_tensor_list -- a list of Tensor(channel * H * W)
    path -- path of saving directory
    filename_list -- a list with a **batch** number of name
    """
    assert(len(img_tensor_list) == len(filename_list))
    for i, filename in enumerate(filename_list):
        save_filename = filename + '.' + img_type
        torchvision.utils.save_image(img_tensor_list[i], os.path.join(path, save_filename))


def show_img(img_tensor):
    """
    img_tensor should be a 3d tensor, standing for RGB
    """
    npimg = img_tensor.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))