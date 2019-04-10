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
        for i, l in enumerate(self.loss):
            loss = l['function'](fake, real)
            effective_loss = l['weight'] * loss
            losses.append(effective_loss)
            if self.use_tensorboard:
                self.tb_logger.add_scalar(l['type'], loss.item(), step)
        loss_sum = sum(losses)
        return loss_sum
