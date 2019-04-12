import argparse

parser = argparse.ArgumentParser(description='No description')

parser.add_argument('--name', type=str, default='unknown')

# Hardware specifications
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=4, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--use_tensorboard', action='store_true', default=True)
parser.add_argument('--num_workers', type=int, default=24)

# Data specifications
# parser.add_argument('--train_data_path', type=str, default='...')
parser.add_argument('--train_HR', type=str, default='/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/HR', help='path to HR image of training set')
parser.add_argument('--train_LR', type=str, default='/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/LR', help='path to LR image of training set')
parser.add_argument('--val_HR', type=str, default='/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALHR', help='path to HR image of validation set')
parser.add_argument('--val_LR', type=str, default='/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALLR', help='path to LR image of validation set')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')

# Model specifications
parser.add_argument('--model', default='RRDB_enhanced')
parser.add_argument('--pre_train_model', type=str, default='...', help='pre-trained model path')
parser.add_argument('--pre_train_optimizer', type=str, default='...', help='pre-trained optimizer path')


# RRDB_enhanced specifications (args name start with 'a')
parser.add_argument('--a_nb', type=int, default=5, help='Number of RRDB in a trunk branch')
parser.add_argument('--a_na', type=int, default=4,  help='Number of attention modules')
parser.add_argument('--a_nf', type=int, default=64, help='Number of channel of the extrated feature by RRDB')
parser.add_argument('--a_dense_attention_modules', action='store_true', help='Only supported when na=3, mainly used for a single experiment here')


# RCAN specifications
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications

parser.add_argument('--val_every', type=int, default=1000, help='do validation per N iter')
parser.add_argument('--save_every', type=int, default=1000, help='save per N iter')
parser.add_argument('--niters', type=int, default=12000, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=80, help='input batch size for training')
parser.add_argument('--save_optimizer', action='store_true')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step_3000_6000',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration, you should specify like: w1*L1+w2*FaceSphere+...')


# Other specifications
parser.add_argument('--print_every', type=int, default=100, help='how many batches to wait before logging training status')

args = parser.parse_args()


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
if args.a_dense_attention_modules and args.a_na != 3:
    raise NotImplementedError('Dense connection for attention module is only available for a_na = 3')
