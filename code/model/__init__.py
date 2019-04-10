import logging
import torch.nn as nn

logger = logging.getLogger('base')

def create_model(args):
    model = args.model

    if model == 'RRDB_enhanced':
        from .RRDB_enhanced import RRDB_enhanced as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(args)
    m.load_state_dict(torch.load(args.pre_train))
    
    if not args.cpu:
        if args.n_GPUs > 0:
            m = m.cuda()
        if args.n_GPUs > 1:
            m = nn.DataParallel(m, device_ids=[i for i in range(args.n_GPUs)])
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m