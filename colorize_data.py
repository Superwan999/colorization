import os

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

import numpy as np
from PIL import Image

from typing import Tuple
import random

from config import load_config


def collate_fn(batch):
    # use the customized collate_fn to filter out bad inputs
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ColorizeData(Dataset):
    def __init__(self, mode, file_list):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToPILImage(),
                                          T.Resize(size=(256, 256)),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize(0.5, 0.5)
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToPILImage(),
                                           T.Resize(size=(256, 256)),
                                           T.ToTensor(),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
        self.file_list = file_list
        self.mode = mode
        if mode == 'basic':
            pass
        else:
            self.input_transform_128 = T.Compose([T.ToPILImage(),
                                                  T.Resize(size=(128, 128), interpolation=Image.LANCZOS),
                                                  T.Grayscale(),
                                                  T.ToTensor(),
                                                  T.Normalize(0.5, 0.5)
                                                  ])
            self.input_transform_64 = T.Compose([T.ToPILImage(),
                                                 T.Resize(size=(64, 64), interpolation=Image.LANCZOS),
                                                 T.Grayscale(),
                                                 T.ToTensor(),
                                                 T.Normalize(0.5, 0.5)
                                                 ])

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.file_list)

    def __getitem__(self, index: int):  # -> Tuple(torch.Tensor, torch.Tensor):
        # Return the input tensor and output tensor for training
        img_name = self.file_list[index]
        image = np.asarray(Image.open(img_name))
        try:
            if self.mode == 'basic':
                input_image, target_image = self.input_transform(image), self.target_transform(image)
            else:
                input_image, input_image_128, input_image_64, target_image = \
                    self.input_transform(image), self.input_transform_128(image), \
                    self.input_transform_64(image), self.target_transform(image)
        except:
            # there are some strange exceptions
            return None
        if self.mode == 'basic':
            return input_image, target_image
        else:
            return input_image, input_image_128, input_image_64, target_image


class ColorizeDataset:
    def __init__(self, args):
        random.seed(args.seed)

        # generate the list contains all data/images
        if args.phase == 'train' and not os.path.isdir(args.path):
            print('For training phase, only data folder is supported')
            raise
        all_img_ids = [os.path.join(args.path, img_id) for img_id in os.listdir(args.path)]
        random.shuffle(all_img_ids)

        # generate training & validating image lists
        if args.val_ratio == 0:
            # Finally use all image to train
            self.train_dataset = ColorizeData(args.mode, all_img_ids)
            self.val_dataset = ColorizeData(args.mode, random.sample(all_img_ids, int(0.3 * len(all_img_ids))))
        else:
            train_size = int(len(all_img_ids) * (1 - args.val_ratio))
            train_ids = all_img_ids[:train_size]
            val_ids = all_img_ids[train_size:]

            # generate training & validating
            self.train_dataset = ColorizeData(args.mode, train_ids)
            self.val_dataset = ColorizeData(args.mode, val_ids)


if __name__ == '__main__':
    args = load_config()
    colorize_dataset = ColorizeDataset(args)
