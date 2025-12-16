import os.path

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from transformers import get_scheduler

from layers.Gaussian import Gaussian
from models.Net import Diffusion
from data_loader.eval_dataset import LytroDataset
from data_loader.pretrain_dataset import TrainDataset
from utils import *

checkpoint = {}


def train(args, diff, train_dataloader, lytro_dataloader, logger):
    now = args.model + ' | ' + time.strftime('%Y-%m-%d|%H:%M:%S', time.localtime())
    tf_logs_dir = os.path.join(args.tf_logs_dir, 'pretrain', now)

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

    T = 1000
    gauss = Gaussian(time_steps=T, device=args.device)

    loss = torch.tensor(0.)
    best_loss = torch.inf
    iter_loss = torch.tensor(0.)

    global checkpoint
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

    for epoch in range(start, args.epochs):
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, ncols=80)
        bar.set_description(f'[{epoch + 1}/{args.epochs}]')
        bar.set_postfix({'loss': f'{loss.item():>5.4f}', })

        improve = ''
        diff.train()
        for i, (S1, S2, GT, fused_scheme) in bar:
            t = torch.randint(0, T, (args.batch_size,), device=args.device).long()
            S1, S2, GT, fused_scheme = S1.to(args.device), S2.to(args.device), GT.to(
                args.device), fused_scheme.to(args.device)
            x_noisy, noise = gauss.forward_diffusion_sample(GT, t)
            noise_pred = diff(x_noisy, t, S1, S2, embeddings=fused_scheme)
            loss = F.l1_loss(noise, noise_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():>5.4f}', })

        scheduler.step()
        iter_loss = iter_loss / len(train_dataloader)

        if iter_loss < best_loss and (epoch + 1) > args.warmup_steps:
            improve = '*'
            best_loss = iter_loss

            best_loss_state = {
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

            torch.save(best_loss_state, os.path.join(args.log_dir, args.model + '_best_loss.ckpt'))

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'[{epoch + 1}/{args.epochs}]')
        logger.info(f'Learning rate: {lr:>5.8f}')
        logger.info(f'loss: {iter_loss:>5.4f}{improve}')
        logger.info(f'best_loss: {best_loss:>5.4f}')

        writer.add_scalar('training/iter_loss', iter_loss, epoch)
        writer.add_scalar('training/lr', lr, epoch)
        iter_loss = torch.tensor(0.0)

        time_dif = get_time_dif(start_time)
        logger.info(f'time usage: {time_dif}')

        checkpoint = {
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

        if (epoch + 1) % 1000 == 0:
            logger.info(f'Saving checkpoint at epoch {epoch + 1}')
            torch.save(checkpoint, os.path.join(args.log_dir, args.model +f"_ep{epoch+1}.ckpt"))

    logger.info("Training completed, starting evaluation.")

    diff.load_state_dict(torch.load(os.path.join(args.log_dir, args.model + '_best_loss.ckpt'))['diff_state_dict'])

    psnr = []
    ssim = []
    Qabf = []
    fused_scheme = torch.tensor([0]).to(args.device)

    norm = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    denorm = denormalizer([0.5, ], [0.5, ])

    for i, (S1, S2, names) in enumerate(lytro_dataloader):
        S1, S2 = S1.to(args.device), S2.to(args.device)
        Y_1, Cb_1, Cr_1 = rgb2ycbcr(S1)
        Y_2, Cb_2, Cr_2 = rgb2ycbcr(S2)
        Y_1, Y_2 = norm(Y_1), norm(Y_2)
        Cb, Cr = fuse_cb_cr(Cb_1, Cr_1, Cb_2, Cr_2)

        fused = gauss.ddpm_sample(diff, Y_1, Y_2, embeddings=fused_scheme, desc=f'Generating {names[0]}')
        # fused = gauss.ddim_sample(diff,Y_1,Y_2,embeddings=fused_scheme,ddim_timesteps=20,desc=f'Generating {names[0]}')
        Y_1, Y_2, fused = denorm(Y_1), denorm(Y_2), denorm(fused)

        Qabf.append(metric_Qabf(Y_1, Y_2, fused))
        psnr.append(metric_psnr(Y_1, Y_2, fused))
        ssim.append(metric_ssim(Y_1, Y_2, fused))

        fused = ycbcr2rgb(fused, Cb, Cr)
        fused = torch.clamp(fused, 0, 1)

        fused = fused.squeeze(0).cpu()
        writer.add_image(f'lytro', fused, global_step=i)

    writer.add_scalar('metric/PSNR', sum(psnr) / len(psnr))
    writer.add_scalar('metric/SSIM', sum(ssim) / len(ssim))
    writer.add_scalar('metric/Qabf', sum(Qabf) / len(Qabf))

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

    diff = Diffusion()
    diff.to(args.device)

    checkpoint = torch.load(os.path.join(args.log_dir, args.model + '_tmp.ckpt')) if args.restart else checkpoint

    logger.info('TRAINING ...')

    try:
        train(args, diff, train_dataloader, lytro_dataloader, logger)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f'Epoch: {checkpoint["epoch"]}')
        logger.error(f'Best loss: {checkpoint["best_loss"]}')
        torch.save(checkpoint, os.path.join(args.log_dir, args.model + '_tmp.ckpt'))
        raise e
    except KeyboardInterrupt:
        logger.error('Training interrupted')
        logger.error(f'Epoch: {checkpoint["epoch"]}')
        logger.error(f'Best loss: {checkpoint["best_loss"]}')
        torch.save(checkpoint, os.path.join(args.log_dir, args.model + '_tmp.ckpt'))
