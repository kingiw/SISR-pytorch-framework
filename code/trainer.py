import utils
import logging
import torch

logger = logging.getLogger('base')


class Trainer():
    def __init__(self, args, train_loader, val_loader, model, loss):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        if args.pre_train_model != "...":
            self.model.load_state_dict(torch.load(args.pre_train_model))
        self.loss = loss

        # Actually I am not sure what would happen if I specify the optimizer and scheduler in main.py
        self.optimizer = utils.make_optimizer(self.args, self.model)
        if args.pre_train_optimizer != "...":
            self.optimizer.load_state_dict(torch.load(args.pre_train_optimizer))
        self.scheduler = utils.make_scheduler(self.args, self.optimizer)
        self.iter = 0

    def train(self):
        self.model.train()
        self.scheduler.step()
        timer_data, timer_model = utils.timer(), utils.timer()
        timer_model.hold()

        while 1:
            for batch, (name, lr, hr) in enumerate(self.train_loader):
                
                if (not self.args.cpu) and self.args.n_GPUs > 0:
                    lr, hr = lr.cuda(), hr.cuda()
                self.iter += 1
                
                self.scheduler.last_epoch = self.iter
                timer_data.hold()
                timer_model.tic()
                
                self.optimizer.zero_grad()
                sr = self.model(lr)

                # loss is the sum of losses
                loss, losses = self.loss(sr, hr, self.iter)
                loss.backward()
                self.optimizer.step()

                timer_model.hold()
                
                if self.iter % self.args.print_every == 0:
                    logger.info('Iter {}, Lr: {:.2e}     {:.1f}s+{:.1f}s'.format(
                        self.iter,
                        self.scheduler.get_lr()[0],
                        timer_model.release(),
                        timer_data.release()
                    ))
                    s = ""
                    for l in losses:
                        s += '{}: {:.4e} '.format(l['type'], l['loss'])
                    s += " Total: {:.4e}".format(loss.item())
                    logger.info(s)
                
                if self.iter % self.args.save_every == 0:
                    logger.info('Saving model and optimizer...')
                    torch.save(self.model.state_dict(), "../experiments/{}/model/{}.pth".format(self.args.name, self.iter))
                    if self.args.save_optimizer:
                        torch.save(self.optimizer.state_dict(), "../experiments/{}/optimizer/{}.pth".format(self.args.name, self.iter))
                
                if self.iter % self.args.val_every == 0:
                    self.test()

                if self.iter > self.args.niters:
                    logger.info('End of trainning')
                    torch.save(self.model.state_dict(), "../experiments/{}/model/latest.pth".format(self.args.name, self.iter))
                    return
                timer_data.tic()
                
            
    def test(self):
        """
        Only support for default batch_size = 1
        I'll update this function later
        """
        self.model.eval()
        logger.info('Evaluating...')
        timer_test = utils.timer()
        timer_test.tic()
        name_list = []
        saved_img = []
        with torch.no_grad():
            # losses = [{'type': xxx, 'loss': xxx, 'weight':xxx}]
            avg_loss = 0
            avg_losses = []
            for i, (name, lr, hr) in enumerate(self.val_loader):
                if (not self.args.cpu) and self.args.n_GPUs > 0:
                    lr, hr = lr.cuda(), hr.cuda()
                sr = self.model(lr)
                loss, losses = self.loss(sr, hr, self.iter, is_train=False)

                if i == 0:
                    avg_losses = losses
                else:
                    for k in range(len(losses)):
                        avg_losses[k]['loss'] += losses[k]['loss']
                avg_loss += loss

                if name[0][:3] == "000":
                    name_list.append("{}_{}_{}".format(name[0], self.args.name, self.iter))
                    saved_img.append(sr[0])
        timer_test.hold()
        s = "\n"
        for l in avg_losses:
            s += '{}: {:.4e} '.format(l['type'], l['loss'].item() / len(self.val_loader))
        s += " Total: {:.4e}".format(avg_loss.item() / len(self.val_loader))
        s += " Time Elapsed: {:.1f}\n".format(timer_test.release())
        logger.info(s)
        utils.save_image(saved_img, '../experiments/{}/results'.format(self.args.name), name_list)