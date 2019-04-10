import torch
import torch.nn as nn
import logging
import os


logger = logging.getLogger('base')

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        
        self.n_GPUs = args.n_GPUs
        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            if not os.path.isdir(os.path.join('../tb_logger/' + args.name)):
                os.mkdir(os.path.join('../tb_logger/' + args.name))
            self.tb_logger = SummaryWriter(log_dir=os.path.join('../tb_logger/' + args.name))

        # self.loss contains the sub-loss function infomation
        self.loss = []
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'FaceSphere':
                from face_sphere import FaceSphereLoss
                loss_function = FaceSphereLoss(args.n_GPUs)
            else:
                NotImplementedError('Loss [{:s}] not recognized.'.format(loss_type))
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })
    
        for l in self.loss:
                logger.info('{:.3f}*{}'.format(l['weight'], l['type']))
             
        
        logger.info('Loss function preparation done.')
    def forward(self, fake, real, step):
        losses = []
        loss_sum = torch.zeros(1)
        for i, l in enumerate(self.loss):
            loss = l['function'](fake, real)
            losses.append({
                'loss': loss,
                'type': l['type'],
                'weight': l['weight']
            })
            loss_sum = l['weight'] * loss + loss_sum
            if self.use_tensorboard:
                self.tb_logger.add_scalar(l['type'], loss.item(), step)
        
        
        return loss_sum, losses
