import os
import random
import time
from layers.Loss_Y import Sobelxy
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf

sobelxy = Sobelxy()


class TrainDataset(Dataset):
    def __init__(self, root="/home/data/Diffusion"):
        super().__init__()

        self.img_size = 128
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ])

        self.MFI = os.path.join(root, 'MFI')
        # self.MEF = os.path.join(root, 'MEF')
        # self.MSRS = os.path.join(root, 'MSRS')

        self.source_1 = []
        self.source_2 = []
        self.target = []
        self.fused_scheme = []

        for file_name in os.listdir(os.path.join(self.MFI, 'NY')):
            S1 = Image.open(os.path.join(self.MFI, 'NY', file_name)).convert('L')
            S2 = Image.open(os.path.join(self.MFI, 'NY_1', file_name)).convert('L')
            target = Image.open(os.path.join(self.MFI, 'NY_2', file_name)).convert('L')
            S1 = self.transform(S1)
            S2 = self.transform(S2)
            target = self.transform(target)
            self.source_1.append(S1)
            self.source_2.append(S2)
            self.target.append(target)
            # self.target.append(os.path.join(self.MFI, 'NY', file_name))
            # self.source_1.append(os.path.join(self.MFI, 'NY_1', file_name))
            # self.source_2.append(os.path.join(self.MFI, 'NY_2', file_name))
            self.fused_scheme.append(0)

        # for file_name in os.listdir(os.path.join(self.MEF, 'NY')):
        #     self.target.append(os.path.join(self.MEF, 'NY', file_name))
        #     self.source_1.append(os.path.join(self.MEF, 'NY_1', file_name))
        #     self.source_2.append(os.path.join(self.MEF, 'NY_2', file_name))
        #     self.fused_scheme.append(1)
        #
        # for file_name in os.listdir(os.path.join(self.MSRS, 'NY')):
        #     self.target.append(os.path.join(self.MSRS, 'NY', file_name))
        #     self.source_1.append(os.path.join(self.MSRS, 'NY_1', file_name))
        #     self.source_2.append(os.path.join(self.MSRS, 'NY_2', file_name))
        #     self.fused_scheme.append(2)



    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        # S1 = self.transform(Image.open(self.source_1[index]))
        # S2 = self.transform(Image.open(self.source_2[index]))
        # target = self.transform(Image.open(self.target[index]))

        S1 = self.source_1[index]
        S2 = self.source_2[index]
        target = self.target[index]

        h, w = S1.shape[1:]
        if h <= self.img_size or w <= self.img_size:
            S1 = tf.resize(S1, [self.img_size, self.img_size])
            S2 = tf.resize(S2, [self.img_size, self.img_size])
            target = tf.resize(target, [self.img_size, self.img_size])
        else:
            h = random.randint(0, h - self.img_size)
            w = random.randint(0, w - self.img_size)

            S1 = tf.crop(S1, h, w, self.img_size, self.img_size)
            S2 = tf.crop(S2, h, w, self.img_size, self.img_size)
            target = tf.crop(target, h, w, self.img_size, self.img_size)
        S1, S2, target = self._data_augment(S1, S2, target)
        return S1, S2, target, self.fused_scheme[index]

    def _data_augment(self, S1, S2, target):
        if random.random() > 0.5:
            S1 = tf.hflip(S1)
            S2 = tf.hflip(S2)
            target = tf.hflip(target)

        if random.random() > 0.5:
            S1 = tf.vflip(S1)
            S2 = tf.vflip(S2)
            target = tf.vflip(target)

        return S1, S2, target



class LytroDataset(Dataset):
    def __init__(self, path="/home/data/Diffusion"):
        super().__init__()
        path = path + '/Lytro'
        self.source_1 = os.path.join(path, 'A_jpg')
        self.source_2 = os.path.join(path, 'B_jpg')
        self.file_list = os.listdir(self.source_1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index])).convert('L')
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index].replace('A', 'B'))).convert('L')
        S1 = self.transform(S1)
        S2 = self.transform(S2)
        # intens = torch.max(S1,S2)
        # grad = torch.max(sobelxy(S1),sobelxy(S2))
        return S1, S2, self.file_list[index].replace('-A', '')


class MEFBDataset(Dataset):
    def __init__(self, path="/home/data/Diffusion"):
        super().__init__()
        path = path + '/MEFB'
        self.path = path
        self.file_list = os.listdir(path)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        image_files = os.listdir(os.path.join(self.path, file))
        S1, S2 = None, None

        for image in image_files:
            if '_A' in image or '_a' in image:
                S1 = Image.open(os.path.join(self.path, file, image)).convert('L')
                S1 = self.to_tensor(S1)
                continue
            if '_B' in image or '_b' in image:
                S2 = Image.open(os.path.join(self.path, file, image)).convert('L')
                S2 = self.to_tensor(S2)
                continue
        # intens = (S1+S2) / 2
        # grad = torch.max(sobelxy(S1),sobelxy(S2))

        return S1, S2, file + '.jpg'


class MSRSDataset(Dataset):
    def __init__(self, path="/home/data/Diffusion"):
        super().__init__()
        path = path + '/MSRS_test'
        self.source_1 = os.path.join(path, 'ir')
        self.source_2 = os.path.join(path, 'vi')
        self.file_list = os.listdir(self.source_1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index])).convert('L')
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index])).convert('L')
        S1 = self.transform(S1)
        S2 = self.transform(S2)
        # intens = torch.max(S1, S2)
        # grad = torch.max(sobelxy(S1), sobelxy(S2))
        return S1, S2, self.file_list[index]


if __name__ == '__main__':
    dataset = TrainDataset()
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(data_loader.__len__())
    for S1, S2, target, f in data_loader:
        print(S1.shape, S2.shape, target.shape, f)
