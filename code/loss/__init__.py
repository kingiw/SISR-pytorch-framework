import torch
import torch.nn as nn
import logging
import os
import ganloss


logger = logging.getLogger('base')

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        
        self.n_GPUs = args.n_GPUs
        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.tb_logger = SummaryWriter(log_dir=os.path.join('../experiments/{}/tb_logger/'.format(args.name)))

        # self.loss contains the sub-loss function infomation
        self.loss = []
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'FaceSphere':
                from .face_sphere import FaceSphereLoss
                loss_function = FaceSphereLoss(args.n_GPUs)
            elif loss_type == 'GanLoss':
                loss_function =  ganloss.GANLoss(args) 
            else:
                NotImplementedError('Loss [{:s}] not recognized.'.format(loss_type))
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                # 'function': loss_function.cuda() if args.n_GPUs > 0 else loss_function
                'function': loss_function
            })
    
        for l in self.loss:
            logger.info('{:.3f}*{}'.format(l['weight'], l['type']))
        
        logger.info('Loss function preparation done.')

    def forward(self, fake, real, step, is_train=True):
        losses = []
        loss_sum = torch.zeros(1)
        if self.n_GPUs > 0:
            loss_sum = loss_sum.cuda()
        for i, l in enumerate(self.loss):
            loss = l['function'](fake, real)
            losses.append({
                'loss': loss,
                'type': l['type'],
                'weight': l['weight']
            })
            loss_sum = l['weight'] * loss + loss_sum
            if self.use_tensorboard:
                self.tb_logger.add_scalar(l['type'] + ("_val" if not is_train else ""), loss.item(), step)
        
        
        return loss_sum, losses
