import os.path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import WarmupLinearSchedule

from layers.Gaussian import Gaussian
from models01.Net01 import Diffusion
from data_loader.data_loader01 import LytroDataset, MEFBDataset, MSRSDataset, TrainDataset
from utils import get_logger, get_time_dif, parse_args, denormalizer, seed_everything, metric_psnr, metric_ssim, \
    metric_Qabf


def train(args, diff, train_dataloader, val_dataloader, logger, checkpoint):
    now = time.strftime('%Y-%m-%d|%H:%M:%S', time.localtime())
    tf_logs_dir = os.path.join(args.tf_logs_dir, args.model, now)

    start = 0
    start_time = time.time()

    no_decay = ['norm.weight', 'bias']
    optimizer_parameters = [
        {'params': [p for n, p in diff.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.05, 'lr': args.lr},
        {'params': [p for n, p in diff.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.lr},
    ]

    optimizer = AdamW(optimizer_parameters, args.lr)
    scheduler = LinearLR(optimizer, 1, 0, args.epoch // 50)

    T = 1000
    gauss = Gaussian(time_steps=T, device=args.device)

    loss = torch.tensor(0.0)
    best_loss = torch.inf
    iter_loss = torch.tensor(0.0)

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

    for epoch in range(start, args.epoch):
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, ncols=120)
        bar.set_description(f'[{epoch + 1}/{args.epoch}]')
        bar.set_postfix({'loss': f'{loss.item():>5.4f}', })

        improve = ''
        diff.train()
        for i, (S1, S2, targe, fused_scheme) in bar:
            t = torch.randint(0, T, (args.batch_size,), device=args.device).long()
            S1, S2, targe, fused_scheme = S1.to(args.device), S2.to(args.device), targe.to(
                args.device), fused_scheme.to(args.device)
            x_noisy, noise = gauss.forward_diffusion_sample(targe, t)
            noise_pred = diff(x_noisy, t, S1, S2, embeddings=fused_scheme)
            loss = F.l1_loss(noise, noise_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():>5.4f}', })

        iter_loss = iter_loss / len(train_dataloader)
        if (epoch + 1) % 50 == 0:
            scheduler.step()

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
            torch.save(state, os.path.join(args.log_dir, args.model + '.ckpt'))

        time_dif = get_time_dif(start_time)

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'[{epoch + 1}/{args.epoch}]')
        logger.info(f'Learning rate: {lr:>5.8f}')
        logger.info(f'loss: {iter_loss:>5.4f}{improve}')
        logger.info(f'best_loss: {best_loss:>5.4f}')
        logger.info(f'time usage: {time_dif}')
        writer.add_scalar('loss/iter_loss', iter_loss, epoch)
        iter_loss = torch.tensor(0.0)

        if (epoch + 1) % 500 == 0:

            psnr = []
            ssim = []
            Qabf = []
            fused_scheme = torch.tensor([0]).to(args.device)
            for i, (S1, S2, _) in enumerate(val_dataloader[0]):
                S1, S2 = S1.to(args.device), S2.to(args.device)
                fused = gauss.ddim_sample(diff, S1, S2, embeddings=fused_scheme, ddim_timesteps=50)
                S1, S2, fused = denorm(S1), denorm(S2), denorm(fused)
                for j, img in enumerate(fused.split(1, dim=0)):
                    writer.add_image(f'epoch_{epoch}/ddim_lytro', img.squeeze(0).cpu(), global_step=i * 5 + j)

                psnr.append(metric_psnr(S1, S2, fused))
                ssim.append(metric_ssim(S1, S2, fused))
                Qabf.append(metric_Qabf(S1, S2, fused))

            writer.add_scalar('lytro/psnr', sum(psnr) / len(psnr), epoch // 500)
            writer.add_scalar('lytro/ssim', sum(ssim) / len(ssim), epoch // 500)
            writer.add_scalar('lytro/Qabf', sum(Qabf) / len(Qabf), epoch // 500)

            psnr = []
            ssim = []
            Qabf = []
            fused_scheme = torch.tensor([1]).to(args.device)
            for i, (S1, S2, _) in enumerate(val_dataloader[1]):
                if i > 20:
                    break
                S1, S2 = S1.to(args.device), S2.to(args.device)
                fused = gauss.ddim_sample(diff, S1, S2, embeddings=fused_scheme, ddim_timesteps=30)
                S1, S2, fused = denorm(S1), denorm(S2), denorm(fused)
                writer.add_image(f'epoch_{epoch}/ddim_mefb', fused.squeeze(0).cpu(), global_step=i)

                psnr.append(metric_psnr(S1, S2, fused))
                ssim.append(metric_ssim(S1, S2, fused))
                Qabf.append(metric_Qabf(S1, S2, fused))

            writer.add_scalar('mefb/psnr', sum(psnr) / len(psnr), epoch // 500)
            writer.add_scalar('mefb/ssim', sum(ssim) / len(ssim), epoch // 500)
            writer.add_scalar('mefb/Qabf', sum(Qabf) / len(Qabf), epoch // 500)



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
    lytro_dataloader = DataLoader(lytro_dataset, batch_size=5, shuffle=False)

    mefb_dataset = MEFBDataset()
    mefb_dataloader = DataLoader(mefb_dataset, batch_size=1, shuffle=True)

    val_dataloader = [lytro_dataloader, mefb_dataloader]

    diff = Diffusion()
    diff.to(args.device)

    checkpoint = torch.load(os.path.join(args.log_dir, args.model + '.ckpt')) if args.restart else None
    logger.info('TRAINING ...')

    train(args, diff, train_dataloader, val_dataloader, logger, checkpoint)
