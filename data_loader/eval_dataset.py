import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf

root = "/home/data/datasets/Diffusion"
# root = "/root/autodl-tmp/Diffusion"



class LytroDataset(Dataset):
    def __init__(self, root=root):
        super().__init__()
        self.path = os.path.join(root, "Lytro")
        self.source_1 = os.path.join(self.path, 'A_jpg')
        self.source_2 = os.path.join(self.path, 'B_jpg')
        self.file_list = os.listdir(self.source_1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index]))
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index].replace('A', 'B')))
        S1 = self.transform(S1)
        S2 = self.transform(S2)
        return S1, S2, self.file_list[index].replace('-A', '')


class MEFBDataset(Dataset):
    def __init__(self, root=root):
        super().__init__()
        self.path = os.path.join(root, "MEFB")
        self.file_list = os.listdir(self.path)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        image_files = os.listdir(os.path.join(self.path, file))
        S1, S2 = None, None

        for image in image_files:
            if '_A' in image or '_a' in image:
                S1 = Image.open(os.path.join(self.path, file, image))
                S1 = self.to_tensor(S1)
                continue
            if '_B' in image or '_b' in image:
                S2 = Image.open(os.path.join(self.path, file, image))
                S2 = self.to_tensor(S2)
                continue

        return S1, S2, file + '.jpg'


class MSRSDataset(Dataset):
    def __init__(self, root=root):
        super().__init__()
        self.path = os.path.join(root, 'MSRS-main','test')
        self.source_1 = os.path.join(self.path, 'ir')
        self.source_2 = os.path.join(self.path, 'vi')
        self.file_list = os.listdir(self.source_1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index]))
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index]))
        S1 = self.transform(S1)
        S2 = self.transform(S2)
        return S1, S2, self.file_list[index]

class CTMRIDataset(Dataset):
    def __init__(self,root=root):
        super().__init__()
        self.path = root + '/Havard/MyDatasets/CT-MRI/test'
        self.CT = os.path.join(self.path, 'CT')
        self.MRI = os.path.join(self.path, 'MRI')
        self.file_list = os.listdir(self.CT)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        S1 = Image.open(os.path.join(self.CT, file_name))
        S2 = Image.open(os.path.join(self.MRI, file_name))
        S1 = self.transform(S1)
        S2 = self.transform(S2)
        return S1, S2, self.file_list[index]


class PETMRIDataset(Dataset):
    def __init__(self, root=root):
        super().__init__()
        self.path = root + '/Havard/MyDatasets/PET-MRI/test'
        self.PET = os.path.join(self.path, 'PET')
        self.MRI = os.path.join(self.path, 'MRI')
        self.file_list = os.listdir(self.PET)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        S1 = Image.open(os.path.join(self.PET, file_name))
        S2 = Image.open(os.path.join(self.MRI, file_name))
        S1 = self.transform(S1)
        S2 = self.transform(S2)
        return S1, S2, self.file_list[index]


if __name__ == '__main__':
    dataset = PETMRIDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(data_loader.__len__())
    for S1, S2,f in data_loader:
        print(S1.shape, S2.shape,f)
