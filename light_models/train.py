import gc
import os
import shutil

import torch
import torch.nn.functional as F
from dataset import *
from torch.utils.data import DataLoader
from config import load_config
from dataset import *
from discriminator import Discriminator
from generator import Generator
from loss import GanLoss


class Trainer:
    def __init__(self, args):
        self.args = args
        dataloader = DataSetLoader(self.args)
        self.trainDataLoader = DataLoader(dataset=dataloader.trainDataSet,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          collate_fn=dataloader.trainDataSet.collate_fn)
        self.validDataLoader = DataLoader(dataset=dataloader.validDataSet,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          collate_fn=dataloader.validDataSet.collate_fn)

        self.Discreaminator = Discriminator()
        self.Generator = Generator()
        self.optimizer_G = torch.optim.Adam(params=self.Generator.parameters(),
                                            lr=args.lr,
                                            betas=(0.5, 0.9))
        self.optimizer_D = torch.optim.Adam(params=self.Discreaminator.parameters(),
                                            lr=args.lr,
                                            betas=(0.5, 0.9))
        self.criterion = GanLoss(self.args)

    def train(self, epoch_index):
        epoch_pixel_loss = 0
        for iteration, batch in enumerate(self.trainDataLoader):
            image, image128, image64, target = batch[0], batch[1], batch[2], batch[3]
            z = torch.rand((len(image), 1, 8, 8))

            pred_img = self.Generator(image, image128, image64, z)

            for d_param in self.Discreaminator.parameters():
                d_param.requires_grad = True

            adv_D_loss, gp_loss, _ = self.criterion('D',
                                                    self.Discreaminator,
                                                    pred_img,
                                                    target,
                                                    epoch_index)
            adv_D_loss = self.args.adv_D_loss_weight * adv_D_loss
            gp_loss = self.args.gp_loss_weight * gp_loss
            D_loss = adv_D_loss + gp_loss
            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()

            for d_param in self.Discreaminator.parameters():
                d_param.requires_grad = False
            adv_G_loss, pixel_loss, perceptual_loss = self.criterion('G',
                                                                     self.Discreaminator,
                                                                     pred_img,
                                                                     target,
                                                                     epoch_index)

            epoch_pixel_loss += pixel_loss.item()
            adv_G_loss = self.args.adv_G_loss_weight * adv_G_loss
            pixel_loss = self.args.pixel_loss_weight * pixel_loss
            perceptual_loss = self.args.perceptual_loss * perceptual_loss

            G_loss = adv_G_loss + pixel_loss + perceptual_loss

            self.optimizer_G.zero_grad()
            G_loss.backward()
            self.optimizer_G.step()

            if iteration % self.args.iter_print == 0:
                show_generator_str = f"===> Generator: " \
                                     f"Epoch[{epoch_index}]" \
                                     f"{iteration}/" \
                                     f"{len(self.trainDataLoader)}: " \
                                     f"adv_G_loss:{adv_G_loss.item()}, " \
                                     f"pixel_loss:{pixel_loss.item()}, " \
                                     f"perceptual_loss: {perceptual_loss.item()}, " \
                                     f"Total_G_loss: {G_loss.item()}"
                print(show_generator_str)

                show_discriminator_str = f"===> Discriminator: " \
                                         f"Epoch[{epoch_index}]" \
                                         f"{iteration}/" \
                                         f"{len(self.trainDataLoader)}: " \
                                         f"adv_D_loss:{adv_D_loss.item()}, " \
                                         f"gp_loss:{gp_loss.item()}, " \
                                         f"Total_G_loss: {D_loss.item()}"
                print(show_discriminator_str)
        print("Train Pixel Loss:", epoch_pixel_loss / len(self.trainDataLoader))

    def validate(self, epoch_index):
        epoch_loss = 0
        self.Generator.eval()
        for iteration, batch in enumerate(self.validDataLoader):
            img, img128, img64, target = batch[0], batch[1], batch[2], batch[3]
            z = torch.rand((len(img), 1, 8, 8))

            with torch.no_grad():
                val_pred_img = self.Generator(img, img128, img64, z)
                val_loss = F.smooth_l1_loss(val_pred_img, target)
                epoch_loss += val_loss.item()
        print("Valid Loss: ", epoch_loss / len(self.validDataLoader))
        return epoch_loss

    def save_checkpoint(self, epoch_id, state, is_best, phase):
        os.makedirs(os.path.join(self.args.save_path, phase), exist_ok=True)
        filename = os.path.join(self.args.save_path, f'{phase}_eppch_{epoch_id}.pth')
        torch.save(state, filename)
        if is_best:
            shutil.copy(filename, self.args.save_path + f'{phase}_epoch_{epoch_id}_best.pth')
        print('checkpoint save to {}'.format(filename))


if __name__ == '__main__':
    gc.collect()
    args = load_config()
    cur_valid_loss = float('inf')
    trainer = Trainer(args)
    is_best = False
    for epoch_id in range(1, args.epochs):
        trainer.train(epoch_id)
        is_best = False

        if epoch_id % args.valid_epoch == 0:
            valid_loss = trainer.validate(epoch_id)
            if valid_loss < cur_valid_loss:
                cur_valid_loss = valid_loss
                is_best = True

            trainer.save_checkpoint(epoch_id,
                                    {
                                        'epoch': epoch_id,
                                        'state_dict':trainer.Discreaminator.state_dict()
                                    },
                                    is_best,
                                    'D')
            trainer.save_checkpoint(epoch_id,
                                    {
                                        'epoch': epoch_id,
                                        'state_dict':trainer.Generator.state_dict()
                                    },
                                    is_best,
                                    'G')
            gc.collect()
    # save last checkpoint
    # save discriminator
    trainer.save_checkpoint(args.epochs,
                            {
                                'epoch': args.epochs,
                                'state_dict': trainer.Discreaminator.state_dict(),
                            },
                            is_best,
                            'D')
    # save generator
    trainer.save_checkpoint(args.epochs,
                            {
                                'epoch': args.epochs,
                                'state_dict': trainer.Generator.state_dict(),
                            },
                            is_best,
                            'G')
