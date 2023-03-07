import os
import shutil

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from colorize_data import ColorizeDataset, collate_fn
from basic_model import Net
from model import Discriminator, Generator
from loss import Loss
from config import load_config

import gc


class Trainer:
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
        if args.mode == 'basic':
            self.model = Net(self.args)
        elif args.mode == 'advanced':
            self.model = Generator(self.args)

        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()

        # # Loss function to use
        # You may also use a combination of more than one loss function
        # or create your own.
        self.criterion = Loss(args)

        # Optimizer
        if self.args.solver == 'adam' or args.solver == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.5, 0.9))
        elif self.args.solver == 'sgd' or self.args.solver == 'SGD':
            self.optimizer = optim.SGD(self.model.paramters(), lr=self.args.lr, momentum=0.9)

        # Scheduler
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=args.gamma)

        # finetune
        if args.reload:
            print('=> loading checkpoint: {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            self.start_epoch_id = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.start_epoch_id = 0

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

            self.model.train()
            self.optimizer.zero_grad()

            # forward
            if self.args.mode == 'basic':
                out_img = self.model(img)
            else:
                out_img = self.model(img, img128, img64)

            # loss
            if args.perceptual_loss:
                loss_pixel_l2, loss_perceptual = self.criterion(out_img, target, epoch_index)
                loss = loss_pixel_l2 + args.perceptual_weight * loss_perceptual
                epoch_loss += loss_pixel_l2.item()
            else:
                loss, _ = self.criterion(out_img, target)
                epoch_loss += loss.item()

            # bp
            loss.backward()
            self.optimizer.step()

            # show training result
            showing_str = '===> ' \
                          'Epoch[{}]({}/{}): ' \
                          'lr: ({:.6f}), ' \
                          'Loss: ({:.4f}), '.format(epoch_index, iteration, len(self.train_dataloader),
                                                    self.optimizer.param_groups[-1]['lr'],
                                                    loss.item()
                                                    )
            if args.perceptual_loss:
                showing_str += ' Perceptual Loss: ({:.4f}), '\
                               ' Pixel Loss: ({:.4f}), '.format(args.perceptual_weight * loss_perceptual.item(),
                                                                loss_pixel_l2.item())

            print(showing_str)
        print("Train Loss: ", epoch_loss / len(self.train_dataloader))

    def validate(self, epoch_index):
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        epoch_loss = 0

        self.model.eval()
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

            with torch.no_grad():
                # forward
                if self.args.mode == 'basic':
                    val_out_img = self.model(img)
                else:
                    val_out_img = self.model(img, img128, img64)
                val_loss = F.smooth_l1_loss(val_out_img, target)
                epoch_loss += val_loss.item()

        print("Valid Loss: ", epoch_loss / len(self.val_dataloader))

        return epoch_loss

    def save_checkpoint(self, epoch_id, state, is_best):
        os.makedirs(self.args.save_path, exist_ok=True)
        filename = os.path.join(self.args.save_path, 'epoch_{}.pth'.format(epoch_id))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.args.save_path + '/' + 'epoch_{}'.format(epoch_id) + '_best.pth')
        print('Checkpoint saved to {}'.format(filename))


# main func: train model
if __name__ == '__main__':
    args = load_config()
    if torch.cuda.is_available() and args.cuda is True:
        args.cuda = True
    else:
        args.cuda = False

    cur_valid_loss = float('inf')
    trainer = Trainer(args)
    is_best = False
    for epoch_id in range(1, args.epochs + 1):
        epoch_id += trainer.start_epoch_id
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
            trainer.save_checkpoint(epoch_id,
                                    {
                                        'epoch': epoch_id,
                                        'state_dict': trainer.model.state_dict(),
                                        'optimizer': trainer.optimizer.state_dict()
                                    },
                                    is_best)
        gc.collect()

        # update learning rate
        # trainer.scheduler.step()

    # save last checkpoint
    trainer.save_checkpoint(args.epochs,
                            {
                                'epoch': args.epochs,
                                'state_dict': trainer.model.state_dict(),
                                'optimizer': trainer.optimizer.state_dict()
                            },
                            is_best)

