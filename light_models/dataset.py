import numpy as np
import random
import torchvision.transforms as T
from PIL import Image
import os
import torch
from torch.utils.data import Dataset


class ColorDataSet(Dataset):
    def __init__(self, files):
        self.files = files
        self.input_transform = T.Compose([T.ToPILImage(),
                                          T.Resize((256, 256)),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize(0.5, 0.5)])
        self.input_transform_128 = T.Compose([T.ToPILImage(),
                                              T.Resize((128, 128)),
                                              T.Grayscale(),
                                              T.ToTensor(),
                                              T.Normalize(0.5, 0.5)])
        self.input_transform_64 = T.Compose([T.ToPILImage(),
                                             T.Resize((64, 64)),
                                             T.Grayscale(),
                                             T.ToTensor(),
                                             T.Normalize(0.5, 0.5)])
        self.target_transform = T.Compose([T.ToPILImage(),
                                           T.Resize((256, 256)),
                                           T.ToTensor(),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index])
        image_array = np.asarray(image)
        input = self.input_transform(image_array)
        input_128 = self.input_transform_128(image_array)
        input_64 = self.input_transform_64(image_array)
        target = self.target_transform(image_array)
        return input, input_128, input_64, target

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


class DataSetLoader:
    def __init__(self, args):
        self.args = args
        self.files = [os.path.join(self.args.path, filename) for filename in os.listdir(self.args.path)]
        random.shuffle(self.files)
        self.valid_list = random.sample(self.files, int(len(self.files) * 0.2))
        self.train_list = list(set(self.files) - set(self.valid_list))
        if self.args.val_ratio > 0:
            self.train_list = random.sample(self.files, int(len(self.files) * (1 - self.args.val_ratio)))
            self.valid_list = list(set(self.files) - set(self.train_list))

        self.trainDataSet = ColorDataSet(self.train_list)
        self.validDataSet = ColorDataSet(self.valid_list)
