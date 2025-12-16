import argparse
import logging
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from kornia.metrics import ssim
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='finetune_udiff', help='model name')  # finetune_udiff_s10_l120_max20 pretrain_udiff_ch1248A6_ep30k
    parser.add_argument('--use_checkpoint', type=str, default='False', choices=['True', 'False'], help='use checkpoint')
    parser.add_argument('--restart', type=str, default='False', choices=['True', 'False'], help='restart')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epochs', default=1000, type=int, help='total epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--warmup_steps', default=10, type=int, help='initial warmup steps')
    parser.add_argument('--device', type=str, default='cuda', help='specify devices')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log path')
    # parser.add_argument('--tf_logs_dir', type=str, default='/root/tf-logs', help='tf_logs path')
    parser.add_argument('--tf_logs_dir', type=str, default='/home/data/tf-logs/UDiff_new', help='tf_logs path')

    return parser.parse_args()

class ColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        log_message = super().format(record)

        # 使用 ANSI 转义序列设置不同的颜色
        if record.levelname == 'ERROR':
            log_message = f'\033[91m{log_message}\033[0m'  # 红色
        elif record.levelname == 'WARNING':
            log_message = f'\033[93m{log_message}\033[0m'  # 黄色
        return log_message

def get_logger(args):
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler_1 = logging.FileHandler(f'{args.log_dir}/{args.model}.log')
    file_handler_1.setLevel(logging.INFO)
    file_handler_1.setFormatter(formatter)
    file_handler_2 = logging.StreamHandler()
    file_handler_2.setLevel(logging.INFO)
    file_handler_2.setFormatter(ColorFormatter('%(message)s'))
    logger.addHandler(file_handler_1)
    logger.addHandler(file_handler_2)
    return logger


def denormalizer(channel_mean=[0.485, 0.456, 0.406], channel_std=[0.229, 0.224, 0.225]):
    '''去归一化'''
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    return transforms.Normalize(mean=MEAN, std=STD)


def get_time_dif(start_time):
    '''获取已使用时间'''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total : {total_num / 10 ** 6}M, Trainable : {trainable_num / 10 ** 6}M')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def rgb2ycbcr(img):
    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    return Y[:, None, ...], Cb[:, None, ...], Cr[:, None, ...]


def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    return torch.cat([R, G, B], 1)


def fuse_cb_cr(Cb1, Cr1, Cb2, Cr2, tao=128, eps=1e-12):
    Cb = (Cb1 * torch.abs(Cb1 - tao) + Cb2 * torch.abs(Cb2 - tao)) / (torch.abs(Cb1 - tao) + torch.abs(Cb2 - tao) + eps)
    Cr = (Cr1 * torch.abs(Cr1 - tao) + Cr2 * torch.abs(Cr2 - tao)) / (torch.abs(Cr1 - tao) + torch.abs(Cr2 - tao) + eps)
    return Cb, Cr


def fuse_seq_cb_cr(Cb, Cr, tao=128, eps=1e-12):
    Cb_n = 0.0
    Cb_d = 0.0
    Cr_n = 0.0
    Cr_d = 0.0
    for b, r in zip(Cb, Cr):
        Cb_n += b * torch.abs(b - tao)
        Cr_n += r * torch.abs(r - tao)

        Cb_d += torch.abs(b - tao)
        Cr_d += torch.abs(r - tao)

    return Cb_n / (Cb_d + eps), Cr_n / (Cr_d + eps)


def metric_ssim(image_A, image_B, image_F):
    return torch.mean(0.5 * ssim(image_A, image_F, 11) + 0.5 * ssim(image_B, image_F, 11))


def metric_psnr(image_A, image_B, image_F):
    MSE_AF = torch.mean((image_A - image_F) ** 2)
    MSE_BF = torch.mean((image_B - image_F) ** 2)
    MSE = (MSE_AF + MSE_BF) / 2
    PSNR = 20 * torch.log10(255 / torch.sqrt(MSE))
    return PSNR


def metric_Qabf(image_A, image_B, image_F):
    gA, aA = Qabf_getArray(image_A)
    gB, aB = Qabf_getArray(image_B)
    gF, aF = Qabf_getArray(image_F)
    QAF = Qabf_getQabf(aA, gA, aF, gF)
    QBF = Qabf_getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = torch.sum(gA + gB)
    nume = torch.sum(QAF * gA + QBF * gB)
    return nume / deno


def Qabf_getArray(img):
    kernelx = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    kernely = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]

    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).to(img)
    weighty = nn.Parameter(data=kernely, requires_grad=False).to(img)

    SAx = F.conv2d(img, weightx, padding=1)
    SAy = F.conv2d(img, weighty, padding=1)

    gA = torch.sqrt(SAx * SAx + SAy * SAy)
    aA = torch.zeros_like(img)
    aA[SAx == 0] = torch.pi / 2
    aA[SAx != 0] = torch.arctan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA


def Qabf_getQabf(aA, gA, aF, gF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    GAF, AAF, QgAF, QaAF, QAF = torch.zeros_like(aA), torch.zeros_like(aA), torch.zeros_like(aA), torch.zeros_like(
        aA), torch.zeros_like(aA)
    GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
    AAF = 1 - torch.abs(aA - aF) / (torch.pi / 2)
    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF
    return QAF


if __name__ == '__main__':
    # x = torch.randn(8, 1, 224, 224)
    # y = torch.randn(8, 1, 224, 224)
    # z = torch.randn(8, 1, 224, 224)
    # print(metric_ssim(x, y, z))
    # print(metric_psnr(x, y, z))
    # print(metric_Qabf(x, y, z))
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    S1 = "/Users/peng_y/Documents/MATLAB/Evaluation-for-Image-Fusion-main/Image/Source-Image/MEFB/ue"
    S2 = "/Users/peng_y/Documents/MATLAB/Evaluation-for-Image-Fusion-main/Image/Source-Image/MEFB/oe"
    fused = "/Users/peng_y/Documents/MATLAB/Evaluation-for-Image-Fusion-main/Image/Algorithm/TC-MoA_MEFB"
    img_name = os.listdir(S1)
    Qabf = []
    for img in tqdm(img_name):
        img_a = Image.open(os.path.join(S1, img)).convert('L')
        img_b = Image.open(os.path.join(S2, img)).convert('L')
        img_f = Image.open(os.path.join(fused, img)).convert('L')
        img_a, img_b, img_f = trans(img_a), trans(img_b), trans(img_f)
        Qabf.append(metric_Qabf(img_a,img_b,img_f))
    print(sum(Qabf)/len(Qabf))