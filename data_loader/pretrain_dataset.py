import os
import random
import time
from time import sleep

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf

# path = "/home/data/NYU-D-100"
path = "/home/data/datasets/Diffusion/MFI_new"
# path= "/root/autodl-tmp/Diffusion/NYU-D-100"
# path= "/root/autodl-tmp/Diffusion/MFI_new"

class TrainDataset(Dataset):
    def __init__(self, root=path):
        super().__init__()

        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ])

        self.source_1 = []
        self.source_2 = []
        self.ground_truth = []

        for file_name in os.listdir(os.path.join(root, 'Y')):
            S1 = os.path.join(root, 'Y_1', file_name)
            S2 = os.path.join(root, 'Y_2', file_name)
            GT = os.path.join(root, 'Y', file_name)

            self.source_1.append(S1)
            self.source_2.append(S2)
            self.ground_truth.append(GT)

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):

        S1 = self.source_1[index]
        S2 = self.source_2[index]
        GT = self.ground_truth[index]

        S1 = Image.open(S1)
        S2 = Image.open(S2)
        GT = Image.open(GT)

        S1 = self.transform(S1)
        S2 = self.transform(S2)
        GT = self.transform(GT)

        h, w = GT.shape[1:]
        h = random.randint(0, h - self.img_size)
        w = random.randint(0, w - self.img_size)

        S1 = tf.crop(S1, h, w, self.img_size, self.img_size)
        S2 = tf.crop(S2, h, w, self.img_size, self.img_size)
        GT = tf.crop(GT, h, w, self.img_size, self.img_size)
        return S1, S2, GT, 0

    def _data_augment(self, S1, S2, GT):
        if random.random() > 0.5:
            S1 = tf.hflip(S1)
            S2 = tf.hflip(S2)
            GT = tf.hflip(GT)

        if random.random() > 0.5:
            S1 = tf.vflip(S1)
            S2 = tf.vflip(S2)
            GT = tf.vflip(GT)

        return S1, S2, GT


if __name__ == '__main__':
    dataset = TrainDataset()
    data_loader = DataLoader(dataset, batch_size=1)
    print(len(data_loader))
    exit(0)
    for S1, S2, target, f in data_loader:
        print(S1.shape, S2.shape, target.shape, f)

    print(data_loader.__len__())
