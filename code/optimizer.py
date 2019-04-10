import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


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
    kwarts['weight_decay']  = args.weight_decay

    return optimizer_function(trainable, **kwargs)

