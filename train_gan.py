import os
import shutil

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from colorize_data import ColorizeDataset, collate_fn
from model import Discriminator, Generator
from loss import LossGAN
from config import load_config

import numpy as np
import gc


class TrainerGAN:
    def __init__(self, args):
        # Define hparams here or load them from a config file
        self.args = args

        # dataset / dataloaders
        self.colorize_dataset = ColorizeDataset(args)
        self.train_dataloader = DataLoader(self.colorize_dataset.train_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=collate_fn,
                                           shuffle=True)
        self.val_dataloader = DataLoader(self.colorize_dataset.val_dataset,
                                         batch_size=self.args.batch_size,
                                         collate_fn=collate_fn,
                                         shuffle=True)

        # model
        self.G = Generator(args)
        self.D = Discriminator(args)
        if args.cuda:
            self.G = torch.nn.DataParallel(self.G).cuda()
            self.D = torch.nn.DataParallel(self.D).cuda()

        # Optimizer
        if self.args.solver == 'adam' or args.solver == 'Adam':
            self.optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                          lr=self.args.lr,
                                          betas=(0.5, 0.9))
            self.optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()),
                                          lr=self.args.lr,
                                          betas=(0.5, 0.9))
        elif self.args.solver == 'sgd' or self.args.solver == 'SGD':
            self.optimizer_G = optim.SGD(filter(lambda p: p.requires_grad, self.G.paramters()),
                                         lr=self.args.lr,
                                         momentum=0.9)
            self.optimizer_D = optim.SGD(filter(lambda p: p.requires_grad, self.D.paramters()),
                                         lr=self.args.lr,
                                         momentum=0.9)

        # # Loss function to use
        # You may also use a combination of more than one loss function
        # or create your own.
        self.criterion = LossGAN(args)

        # Scheduler
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=args.gamma)

        # finetune
        if args.reload:
            # load G model
            print('=> loading checkpoint: {}'.format(args.resume_G))
            checkpoint_G = torch.load(args.resume_G)
            self.G.load_state_dict(checkpoint_G['state_dict'], strict=True)
            self.optimizer_G.load_state_dict(checkpoint_G['optimizer'])
            # load D model
            print('=> loading checkpoint: {}'.format(args.resume_D))
            checkpoint_D = torch.load(args.resume_D)
            self.D.load_state_dict(checkpoint_D['state_dict'], strict=True)
            self.optimizer_D.load_state_dict(checkpoint_D['optimizer'])

    def train(self, epoch_index):
        epoch_loss = 0
        # train loop
        # lr = adjust_learning_rate(self.optimizer, gamma, epoch_index)
        for iteration, batch in enumerate(self.train_dataloader):
            if self.args.mode == 'basic':
                img, target = batch[0], batch[1]
            else:
                img, img128, img64, target = batch[0], batch[1], batch[2], batch[3]
            if self.args.cuda:
                if self.args.mode == 'basic':
                    img, target = img.cuda(), target.cuda()
                else:
                    img, img128, img64, target = img.cuda(), img128.cuda(), img64.cuda(), target.cuda()

            # create noise vector
            # z = torch.FloatTensor(np.random.uniform(-1, 1, (len(img), args.zdim)))
            z = torch.rand((len(img), 1, 8, 8))
            if self.args.cuda:
                z = z.cuda()

            # generate image
            pred_img = self.G(img, img128, img64, z)

            # optimize Discriminator
            for d_parm in self.D.parameters():
                d_parm.requires_grad = True
            adv_D_loss, gp_loss, _ = self.criterion('D', self.D, pred_img, target, epoch_index)
            D_loss = args.adv_D_loss_weight * adv_D_loss + args.gp_loss_weight * gp_loss

            # show training result of Discriminator
            showing_discriminator_str = '===> Discriminator: ' \
                                        'Epoch[{}]({}/{}): ' \
                                        'lr: ({:.6f}), ' \
                                        'adv_D_loss: ({:.6f}), ' \
                                        'gp_loss: ({:.6f}), ' \
                                        'Loss: ({:.6f}), '.format(epoch_index, iteration, len(self.train_dataloader),
                                                                  self.optimizer_D.param_groups[-1]['lr'],
                                                                  args.adv_D_loss_weight * adv_D_loss.item(),
                                                                  args.gp_loss_weight * gp_loss.item(),
                                                                  D_loss.item()
                                                                  )
            print(showing_discriminator_str)

            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()

            # optimize Generator
            for d_parm in self.D.parameters():
                d_parm.requires_grad = False
            adv_G_loss, pixel_loss, perceptual_loss = self.criterion('G', self.D, pred_img, target, epoch_index)
            G_loss = args.adv_G_loss_weight * adv_G_loss + \
                     args.pixel_loss_weight * pixel_loss + \
                     args.perceptual_loss_weight * perceptual_loss

            epoch_loss += pixel_loss.item()

            self.optimizer_G.zero_grad()
            G_loss.backward()
            self.optimizer_G.step()

            # show training result of Generator
            showing_generator_str = '===> Generator: ' \
                                    'Epoch[{}]({}/{}): ' \
                                    'lr: ({:.6f}), ' \
                                    'adv_G_loss: ({:.6f}), ' \
                                    'pixel_loss: ({:.6f}), ' \
                                    'percpetual_loss: ({:.6f}), ' \
                                    'Loss: ({:.6f}), '.format(epoch_index, iteration, len(self.train_dataloader),
                                                              self.optimizer_G.param_groups[-1]['lr'],
                                                              args.adv_G_loss_weight * adv_G_loss.item(),
                                                              args.pixel_loss_weight * pixel_loss.item(),
                                                              args.perceptual_loss_weight * perceptual_loss.item(),
                                                              G_loss.item()
                                                              )
            print(showing_generator_str)
        print("Train Pixel Loss: ", epoch_loss / len(self.train_dataloader))

    def validate(self, epoch_index):
        # Determine your evaluation metrics on the validation dataset.
        epoch_loss = 0

        self.G.eval()
        for iteration, batch in enumerate(self.val_dataloader):
            if self.args.mode == 'basic':
                img, target = batch[0], batch[1]
            else:
                img, img128, img64, target = batch[0], batch[1], batch[2], batch[3]
            if self.args.cuda:
                if self.args.mode == 'basic':
                    img, target = img.cuda(), target.cuda()
                else:
                    img, img128, img64, target = img.cuda(), img128.cuda(), img64.cuda(), target.cuda()

            # z: noise
            # z = torch.FloatTensor(np.random.uniform(-1, 1, (len(img), args.zdim)))
            z = torch.rand((len(img), 1, 8, 8))
            if self.args.cuda:
                z = z.cuda()

            with torch.no_grad():
                # forward
                val_pred_img = self.G(img, img128, img64, z)
                val_loss = F.smooth_l1_loss(val_pred_img, target)
                epoch_loss += val_loss.item()

        print("Valid Loss: ", epoch_loss / len(self.val_dataloader))

        return epoch_loss

    def save_checkpoint(self, epoch_id, state, is_best, phase):
        os.makedirs(self.args.save_path, exist_ok=True)
        filename = os.path.join(self.args.save_path, phase + '_epoch_{}.pth'.format(epoch_id))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.args.save_path + '/' + phase + '_epoch_{}'.format(epoch_id) + '_best.pth')
        print('Checkpoint saved to {}'.format(filename))


# main func: train model
if __name__ == '__main__':
    gc.collect()
    args = load_config()
    if torch.cuda.is_available() and args.cuda is True:
        args.cuda = True
    else:
        args.cuda = False

    cur_valid_loss = float('inf')
    trainer = TrainerGAN(args)
    is_best = False
    for epoch_id in range(1, args.epochs + 1):
        # train
        trainer.train(epoch_id)
        is_best = False

        # validation
        if epoch_id % args.valid_epoch == 0:
            valid_loss = trainer.validate(epoch_id)
            if valid_loss < cur_valid_loss:
                cur_valid_loss = valid_loss
                is_best = True
            # save checkpoint
            # save discriminator
            trainer.save_checkpoint(epoch_id,
                                    {
                                        'epoch': epoch_id,
                                        'state_dict': trainer.D.state_dict(),
                                        'optimizer': trainer.optimizer_D.state_dict()
                                    },
                                    is_best,
                                    'D')
            # save generator
            trainer.save_checkpoint(epoch_id,
                                    {
                                        'epoch': epoch_id,
                                        'state_dict': trainer.G.state_dict(),
                                        'optimizer': trainer.optimizer_G.state_dict()
                                    },
                                    is_best,
                                    'G')
            gc.collect()
        # update learning rate
        # trainer.scheduler.step()

    # save last checkpoint
    # save discriminator
    trainer.save_checkpoint(args.epochs,
                            {
                                'epoch': args.epochs,
                                'state_dict': trainer.D.state_dict(),
                                'optimizer': trainer.optimizer_D.state_dict()
                            },
                            is_best,
                            'D')
    # save generator
    trainer.save_checkpoint(args.epochs,
                            {
                                'epoch': args.epochs,
                                'state_dict': trainer.G.state_dict(),
                                'optimizer': trainer.optimizer_G.state_dict()
                            },
                            is_best,
                            'G')

