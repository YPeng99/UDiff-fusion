import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf

max_images = 100


root = "/home/data/datasets"
# root = "/root/autodl-tmp"
class TrainDataset(Dataset):
    def __init__(self, path=os.path.join(root,'Diffusion')):
        super().__init__()
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ])

        self.MFI = os.path.join(path, 'MFI_new')
        self.MEF = os.path.join(path, 'MEF')
        self.MSRS = os.path.join(path, 'MSRS-main/train')
        self.CT_MRI = os.path.join(path, 'Havard/MyDatasets/CT-MRI/train')
        self.PET_MRI = os.path.join(path, 'Havard/MyDatasets/PET-MRI/train')

        self.source_1 = []
        self.source_2 = []
        self.fused_scheme = []


        for file_name in os.listdir(os.path.join(self.MFI, 'Y_1'))[0:max_images]:
            self.source_1.append(os.path.join(self.MFI, 'Y_1', file_name))
            self.source_2.append(os.path.join(self.MFI, 'Y_2', file_name))
            self.fused_scheme.append(0)

        for file_name in os.listdir(os.path.join(self.MEF, 'Y_1'))[0:max_images]:
            self.source_1.append(os.path.join(self.MEF, 'Y_1', file_name))
            self.source_2.append(os.path.join(self.MEF, 'Y_2', file_name))
            self.fused_scheme.append(1)

        for file_name in os.listdir(os.path.join(self.MSRS, 'Y_1'))[0:max_images]:
            self.source_1.append(os.path.join(self.MSRS, 'Y_1', file_name))
            self.source_2.append(os.path.join(self.MSRS, 'Y_2', file_name))
            self.fused_scheme.append(2)

        for file_name in os.listdir(os.path.join(self.CT_MRI, 'Y_1'))[0:max_images]:
            self.source_1.append(os.path.join(self.CT_MRI, 'Y_1', file_name))
            self.source_2.append(os.path.join(self.CT_MRI, 'Y_2', file_name))
            self.fused_scheme.append(3)

        for file_name in os.listdir(os.path.join(self.PET_MRI, 'Y_1'))[0:max_images]:
            self.source_1.append(os.path.join(self.PET_MRI, 'Y_1', file_name))
            self.source_2.append(os.path.join(self.PET_MRI, 'Y_2', file_name))
            self.fused_scheme.append(4)


    def __len__(self):
        return len(self.fused_scheme)

    def __getitem__(self, index):
        S1 = self.transform(Image.open(self.source_1[index]))
        S2 = self.transform(Image.open(self.source_2[index]))


        h, w = S1.shape[1:]
        h = random.randint(0, h - self.img_size)
        w = random.randint(0, w - self.img_size)

        S1 = tf.crop(S1, h, w, self.img_size, self.img_size)
        S2 = tf.crop(S2, h, w, self.img_size, self.img_size)
        S1, S2 = self._data_augment(S1, S2)
        return S1, S2, self.fused_scheme[index]

    def _data_augment(self, S1, S2):
        if random.random() > 0.5:
            S1 = tf.hflip(S1)
            S2 = tf.hflip(S2)

        if random.random() > 0.5:
            S1 = tf.vflip(S1)
            S2 = tf.vflip(S2)

        return S1, S2




if __name__ == '__main__':
    dataset = TrainDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(data_loader.__len__())
    for S1, S2,f in data_loader:
        print(S1.shape, S2.shape,f)
