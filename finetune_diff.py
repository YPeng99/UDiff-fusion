import os.path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler
from data_loader.test_dataset import *
from data_loader.finetune_dataset import TrainDataset
from models.Net import Diffusion
from utils import get_logger, get_time_dif, parse_args, denormalizer, seed_everything, metric_psnr, metric_ssim, \
    metric_Qabf

from layers.Gaussian import Gaussian
from layers.Loss_Y import FusionLoss


def train(args, diff, train_dataloader, logger, checkpoint):
    now = args.model + ' | ' + time.strftime('%Y-%m-%d|%H:%M:%S', time.localtime())
    tf_logs_dir = os.path.join(args.tf_logs_dir, 'finetune', now)

    start = 0
    start_time = time.time()

    no_decay = ['block.0.weight', 'norm.weight', 'bias']
    optimizer_parameters = [
        {'params': [p for n, p in diff.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-4, 'lr': args.lr},
        {'params': [p for n, p in diff.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0., 'lr': args.lr},
    ]

    optimizer = AdamW(optimizer_parameters, args.lr)
    scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.epochs)

    loss = torch.tensor(0.0)
    best_loss = torch.inf
    iter_loss = torch.tensor(0.0)
    criterion = FusionLoss()
    criterion.to(args.device)

    denorm = denormalizer([0.5, ], [0.5, ])
    if checkpoint:
        start = checkpoint['epoch']
        diff.load_state_dict(checkpoint['diff_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        random.setstate(checkpoint['py_state'])
        np.random.set_state(checkpoint['np_state'])
        torch.set_rng_state(checkpoint['torch_cpu_state'])
        torch.cuda.set_rng_state_all(checkpoint['torch_gpu_state'])
        best_loss = checkpoint['best_loss']
        tf_logs_dir = checkpoint['tf_logs_dir']

    writer = SummaryWriter(str(tf_logs_dir))

    T = 1000
    gauss = Gaussian(time_steps=T, device=args.device)
    ddim_timesteps = 20

    c = T // ddim_timesteps
    ddim_timestep_seq = torch.arange(0, T, c)
    ddim_timestep_seq = ddim_timestep_seq.type(torch.int64) + 1
    ddim_timestep_prev_seq = torch.cat([torch.tensor([0]), ddim_timestep_seq[:-1]])

    for epoch in range(start, args.epochs):

        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, ncols=120,
                   dynamic_ncols=False)
        bar.set_description(f'[{epoch + 1}/{args.epochs}]')
        bar.set_postfix({'loss': f'{loss.item():>5.4f}', })

        improve = ''
        diff.train()
        for i, (S1, S2, fused_scheme) in bar:
            S1, S2, fused_scheme = S1.to(args.device), S2.to(args.device), fused_scheme.to(args.device)
            sample_images = torch.randn_like(S1)

            first_step = True
            for i in reversed(range(0, ddim_timesteps)):
                t = torch.full((sample_images.shape[0],), int(ddim_timestep_seq[i]), device=args.device,
                               dtype=torch.long)
                prev_t = torch.full((sample_images.shape[0],), int(ddim_timestep_prev_seq[i]), device=args.device,
                                    dtype=torch.long)

                alpha_cumprod_t = gauss.get_index_from_list(gauss.alphas_cumprod, t, sample_images.shape)
                alpha_cumprod_t_prev = gauss.get_index_from_list(gauss.alphas_cumprod, prev_t, sample_images.shape)

                pred_noise = diff(sample_images, t, S1, S2, embeddings=fused_scheme,first_step=first_step)

                pred_x0 = (sample_images - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(
                    alpha_cumprod_t)
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

                # 设置 loss
                loss = criterion(S1, S2, x_prev, fused_scheme)
                iter_loss += loss.item()
                difference = F.l1_loss(x_prev.detach(), sample_images.detach()) * ddim_timesteps

                bar.set_postfix({'loss': f'{loss.item() :>5.4f}', 'difference': f'{difference :>5.4f}'})

                loss = loss * difference
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                sample_images = x_prev.detach()

                # sample_images = x_prev
                # first_step = False

            # 设置 loss
            # loss = criterion(S1, S2, sample_images, fused_scheme)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # bar.set_postfix({'loss': f'{loss.item() :>5.4f}'})
            # iter_loss += loss.item()

        scheduler.step()
        iter_loss = iter_loss / ddim_timesteps / len(train_dataloader)
        # iter_loss /= len(train_dataloader)

        if iter_loss < best_loss:
            improve = '*'
            best_loss = iter_loss
            state = {
                'epoch': epoch + 1,
                'diff_state_dict': diff.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'py_state': random.getstate(),
                'np_state': np.random.get_state(),
                'torch_cpu_state': torch.get_rng_state(),
                'torch_gpu_state': torch.cuda.get_rng_state_all(),
                'best_loss': best_loss,
                'tf_logs_dir': tf_logs_dir
            }
            if epoch >= args.warmup_steps:
                torch.save(state, os.path.join(args.log_dir, args.model + '.ckpt'))

        time_dif = get_time_dif(start_time)

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'[{epoch + 1}/{args.epochs}]')
        logger.info(f'Learning rate: {lr:>5.8f}')
        logger.info(f'loss: {iter_loss:>5.4f}{improve}')
        logger.info(f'best_loss: {best_loss:>5.4f}')
        logger.info(f'time usage: {time_dif}')
        writer.add_scalar('training/iter_loss', iter_loss, epoch)
        writer.add_scalar('training/lr', lr, epoch)
        iter_loss = torch.tensor(0.0)

    writer.close()


if __name__ == '__main__':
    # 固定随机种子
    seed_everything(415)

    # args预处理
    args = parse_args()
    args.use_checkpoint = True if args.use_checkpoint == 'True' else False
    args.restart = True if args.restart == 'True' else False

    # 日志
    logger = get_logger(args)
    logger.info('PARAMETER ...')
    logger.info(args)

    now = time.strftime('%Y-%m-%d|%H:%M:%S', time.localtime())
    tf_logs_dir = os.path.join(args.tf_logs_dir, args.model, now)

    logger.info('LOADING ...')
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  prefetch_factor=4, persistent_workers=True, drop_last=True)

    lytro_dataset = LytroDataset()
    lytro_dataloader = DataLoader(lytro_dataset, batch_size=1, shuffle=False)

    mefb_dataset = MEFBDataset()
    mefb_dataloader = DataLoader(mefb_dataset, batch_size=1, shuffle=False)

    msrs_dataset = MSRSDataset()
    msrs_dataloader = DataLoader(msrs_dataset, batch_size=1, shuffle=False)

    ct_mri_dataset = CTMRIDataset()
    ct_mri_dataloader = DataLoader(ct_mri_dataset, batch_size=1, shuffle=False)

    pet_mri_dataset = PETMRIDataset()
    pet_mri_dataloader = DataLoader(pet_mri_dataset, batch_size=1, shuffle=False)

    val_dataloader = [lytro_dataloader, mefb_dataloader, msrs_dataloader, ct_mri_dataloader, pet_mri_dataloader]

    diff = Diffusion()
    state_dict = torch.load(os.path.join(args.log_dir,'tmp', 'pretrain_udiff_ch1248A6_ep30k_best_loss.ckpt'))['diff_state_dict']
    diff.load_state_dict(state_dict, strict=True)

    for k, v in diff.named_parameters():
        if '.embed_q.' not in k and '.proj_q.' not in k:
            v.requires_grad = False
        # if '.embed_q.' not in k:
        #     v.requires_grad = False
        else:
            v.requires_grad = True
    diff.to(args.device)

    checkpoint = torch.load(os.path.join(args.log_dir, args.model + '.ckpt')) if args.restart else None
    logger.info('TRAINING ...')

    train(args, diff, train_dataloader, logger, checkpoint)
