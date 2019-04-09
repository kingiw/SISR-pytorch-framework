import os
import torchvision


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

