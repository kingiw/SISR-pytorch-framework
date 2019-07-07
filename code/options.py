import argparse

parser = argparse.ArgumentParser(description='No description')

parser.add_argument('--name', type=str, default='unknown')
parser.add_argument('--experiment_path', type=str, default='...')

# Hardware specifications
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=4, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--use_tensorboard', action='store_true', default=True)
parser.add_argument('--num_workers', type=int, default=24)

# Data specifications
# parser.add_argument('--train_data_path', type=str, default='...')
parser.add_argument('--train_HR', type=str, default='/BIGDATA1/nsccgz_yfdu_5/lzh/IAA/ffhq/train/hr', help='path to HR image of training set')
parser.add_argument('--train_LR', type=str, default='/BIGDATA1/nsccgz_yfdu_5/lzh/IAA/ffhq/train/lr', help='path to LR image of training set')
parser.add_argument('--val_HR', type=str, default='/BIGDATA1/nsccgz_yfdu_5/lzh/IAA/ffhq/val/hr', help='path to HR image of validation set')
parser.add_argument('--val_LR', type=str, default='/BIGDATA1/nsccgz_yfdu_5/lzh/IAA/ffhq/val/lr', help='path to LR image of validation set')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')

# Model specifications
parser.add_argument('--model', default='RRDB_enhanced', help="RRDB_enhanced | RCAN_enhanced | SRFBN")
parser.add_argument('--pre_train_model', type=str, default='...', help='pre-trained model path')
parser.add_argument('--pre_train_optimizer', type=str, default='...', help='pre-trained optimizer path')


# RRDB_enhanced and RRDB_Net specifications (args name start with 'a')
parser.add_argument('--a_nb', type=int, default=5, help='Number of RRDB in a trunk branch')
parser.add_argument('--a_na', type=int, default=4,  help='Number of attention modules')
parser.add_argument('--a_nf', type=int, default=64, help='Number of channel of the extrated feature by RRDB')
parser.add_argument('--a_dense_attention_modules', action='store_true')
parser.add_argument('--a_ca', action='store_true', help='Add Channel Attention mechanism(noted in paper of RCAN) in RRDB')
# parser.add_argument('--old', action='store_true', help='This option is used for compatibility of the pretrain model in old version.')


# RCAN_enhanced and RCAN specifications (args name start with 'b')
parser.add_argument('--b_n_resgroups', type=int, default=6)
parser.add_argument('--b_n_resblocks', type=int, default=8)
parser.add_argument('--b_n_feats', type=int, default=64)
parser.add_argument('--b_na', type=int, default=3)
parser.add_argument('--b_dense_attention_modules', action='store_true')

# SRFBN specifications (args name start with 'c')
parser.add_argument('--c_nf', type=int, default=64, help='Number of features')
parser.add_argument('--c_ns', type=int, default=4, help='Number of steps')
parser.add_argument('--c_ng', type=int, default=4, help='Number of groups')


# Configure on stage of Upsampling

# Instance Normalization is banned
# parser.add_argument('--use_in', action="store_true", help="Use Instance Normalization at the phase of upsampling.")

# Training specifications

parser.add_argument('--val_every', type=int, default=2500, help='do validation per N iter')
parser.add_argument('--save_every', type=int, default=2500, help='save per N iter')
parser.add_argument('--niters', type=int, default=12000, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
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
parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration, you should specify like: w1*L1+w2*FaceSphere+w3*GanLoss')

# GAN related
parser.add_argument('--gan_type', type=str, default='ragan')
parser.add_argument('--discriminator', type=str, default='discriminator_vgg_128', help='discriminator model, only used when GanLoss is specified')
parser.add_argument('--pretrained_netD', type=str, default=None, help='path of a pretrained discriminator')
parser.add_argument('--weight_decay_D', type=float, default=0)
parser.add_argument('--beta1_D', type=float, default=0.9)
parser.add_argument('--beta2_D', type=float, default=0.99)
parser.add_argument('--lr_D', type=float, default=1e-4)
parser.add_argument('--save_D_every', type=int, default=2500, help='save discriminator every N step')
parser.add_argument('--save_D_path', type=str, default='...')



# Other specifications
parser.add_argument('--print_every', type=int, default=500, help='how many batches to wait before logging training status')

# Test
parser.add_argument('--test_batch_size', type=int, default=80)
parser.add_argument('--test_src', type=str, default='/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/LR')
# parser.add_argument('--test_src', type=str, default='/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALLR_Small')
parser.add_argument('--test_dest', type=str, default='/GPUFS/nsccgz_yfdu_16/lzh/FaceSR/SISR-pytorch-framework/test')


args = parser.parse_args()


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False