import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import discriminator

class GANLoss(nn.Module):
    def __init__(self, args):
        # By default, discriminator will be updated every step
        if args.discriminator == 'discriminator_vgg_128':
            self.netD = discriminator.Discriminator_VGG_128(in_nc=3, nf=64)
        else:
            raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(args.discriminator))

        if args.pretrained_netD is not None:
            self.netD.load_state_dict(torch.load(args.pretrained_netD))
        if not args.cpu:
            self.netD = self.netD.to(self.device)
        if args.n_GPUs > 1:
            self.netD = DataParallel(self.netD)


        self.save_D_every = args.save_D_every
        if args.save_D_path == '...':
            self.save_D_path = "../experiments/{}/model/".format(args.name)
        else:
            self.save_D_path = args.save_D_path

        # Loss Type
        # Loss 接收两个参数 input和target
        # input是由Discriminator提取得到的特征向量
        # target表示input的真/假的取反, 即假如input是真实的，则target是False；假如input是假的，则target是True
        # 我们希望input是由假图输入到discriminator得到的特征，则input应该是尽可能多0。反之输入真图到discriminator得到的特征应该尽可能多1。
        # 简单来说就是用一个loss判定discriminator给出的判断是否远离了事实。
        # 在gan/ragan中，target是一个和input相同size的全0or全1的向量
        # 在wgan-gp中，target就是True或False
        self.gan_type = args.gan_type
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))


        # Optimizer 
        self.optimizer_D = torch.optim.Adam(
            params = self.netD.parameters(),
            lr = args.lr_D,
            betas = (args.beta1_D, args.beta2_D),
            weight_decay = args.weight_decay_D
        )

        

    def get_target_label(self, input, target_is_real):
        if target_is_real:
            return torch.empty_like(input).fill_(1.0) # real_label_val
        else:
            return torch.empty_like(input).fill_(0.0) # fake_label_val

    def update_D(self, loss, step):
        # 根据计算的loss来更新一下D
        
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()
        
        if step % self.save_D_every == 0:
            torch.save(self.netD.state_dict(), "{}/{}.pth".format(self.save_D_path, step))
            print("Discriminatoro saved.")

        for p in self.netD.parameters():
            p.requires_grad = False

    def forward(self, fake, real, step=0, is_train=False):
        # 计算loss值，GAN的LOSS计算是根据当前的输入判断真假
        self.netD.train() if is_train else self.netD.eval()

        pred_d_real = self.netD(self.real)
        pred_d_fake = self.netD(self.fake)

        if self.gan_type == 'gan':
            target_real = self.get_target_label(pred_d_fake, True)
            target_fake = self.get_target_label(pred_d_real, False)
            loss = (self.loss(pred_d_fake, target_real) + self.loss(pred_d_real, target_fake)) / 2
        elif self.gan_type == 'ragan':
            target_real = self.get_target_label(pred_d_fake, True)
            target_fake = self.get_target_label(pred_d_real, False)
            loss = (
                self.loss(pred_d_fake - torch.mean(pred_d_real), target_real) + 
                self.loss(pred_d_real - torch.mean(pred_d_fake), target_fake)
            ) / 2

        if is_train:
            self.update_D(loss, step)
        return loss

