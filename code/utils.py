import os
import torchvision
import matplotlib.pyplot as plt 
import numpy as np
import logging
import os
from datetime import datetime

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


def save_image(img_tensor, path, filename_list, img_type='jpg'):
    """
    img_tensor -- batch * channel * H * W
    path -- path of saving directory
    filename_list -- a list with a **batch** number of name
    """
    if len(img_tensor.shape) == 3: # No batch dimension
        img_tensor = img_tensor.unsqueeze(0)
    assert(img_tensor.shape[0] == len(filename_list))
    for i, filename_list in enumerate(filename_list):
        save_filename = filename + '.' + img_type
        torchvision.utils.save_image(img_tensor[0], os.path.join(path, save_filename))


def show_img(img_tensor):
    """
    img_tensor should be a 3d tensor, standing for RGB
    """
    npimg = img_tensor.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))