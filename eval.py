import os.path
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torch.nn.functional as F
from utils import parse_args
from data_loader.eval_dataset import *
from layers.Gaussian import Gaussian
from models.Net import Diffusion
from utils import *

if __name__ == '__main__':
    # args预处理
    args = parse_args()
    args.use_checkpoint = True if args.use_checkpoint == 'True' else False
    args.restart = True if args.restart == 'True' else False

    save_path = "./results/" + args.model
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'lytro'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'mefb'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'msrs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ct_mri'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet_mri'), exist_ok=True)

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

    diff = Diffusion()
    state_dict = torch.load(os.path.join(args.log_dir, 'pretrain_udiff_ch1248A6_ep30k_best_loss.ckpt'))['diff_state_dict']
    # state_dict = torch.load(os.path.join(args.log_dir, args.model + '.ckpt'))['diff_state_dict']

    diff.load_state_dict(state_dict, strict=True)
    diff.to(args.device)

    to_pil = transforms.ToPILImage()
    to_gray = transforms.Grayscale()
    norm = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    denorm = denormalizer([0.5, ], [0.5, ])
    now = args.model + ' | ' + time.strftime('%Y-%m-%d|%H:%M:%S', time.localtime())
    tf_logs_dir = os.path.join(args.tf_logs_dir, 'eval', now)
    writer = SummaryWriter(tf_logs_dir)

    T = 1000
    gauss = Gaussian(time_steps=T, device=args.device)

    print("Generating Lytro ...")

    psnr = []
    ssim = []
    Qabf = []
    fused_scheme = torch.tensor([0]).to(args.device)
    for i, (S1, S2, names) in enumerate(lytro_dataloader):
        S1, S2 = S1.to(args.device), S2.to(args.device)
        Y_1, Cb_1, Cr_1 = rgb2ycbcr(S1)
        Y_2, Cb_2, Cr_2 = rgb2ycbcr(S2)
        Y_1, Y_2 = norm(Y_1), norm(Y_2)
        Cb, Cr = fuse_cb_cr(Cb_1, Cr_1, Cb_2, Cr_2)

        # fused = gauss.ddim_sample(diff, Y_1, Y_2, embeddings=fused_scheme, ddim_timesteps=20,
        #                           desc=f'Generating {names[0]}')
        fused = gauss.ddpm_sample(diff, Y_1, Y_2, embeddings=fused_scheme, desc=f'Generating {names[0]}')
        Y_1, Y_2, fused = denorm(Y_1), denorm(Y_2), denorm(fused)

        psnr.append(metric_psnr(Y_1, Y_2, fused))
        ssim.append(metric_ssim(Y_1, Y_2, fused))
        Qabf.append(metric_Qabf(Y_1, Y_2, fused))

        fused = ycbcr2rgb(fused, Cb, Cr)
        fused = torch.clamp(fused, 0, 1)

        fused = fused.squeeze(0).cpu()
        writer.add_image(f'lytro', fused, global_step=i)
        fused = to_pil(fused)
        fused.save(os.path.join(save_path, 'lytro', names[0]))

    writer.add_scalar('psnr/0_lytro', sum(psnr) / len(psnr))
    writer.add_scalar('ssim/0_lytro', sum(ssim) / len(ssim))
    writer.add_scalar('Qabf/0_lytro', sum(Qabf) / len(Qabf))

    exit(0)

    print("\nGenerating MEFB ...")

    psnr = []
    ssim = []
    Qabf = []
    fused_scheme = torch.tensor([1]).to(args.device)
    for i, (S1, S2, names) in enumerate(mefb_dataloader):
        S1, S2 = S1.to(args.device), S2.to(args.device)

        Y_1, Cb_1, Cr_1 = rgb2ycbcr(S1)
        Y_2, Cb_2, Cr_2 = rgb2ycbcr(S2)
        Y_1, Y_2 = norm(Y_1), norm(Y_2)
        Cb, Cr = fuse_cb_cr(Cb_1, Cr_1, Cb_2, Cr_2)

        # fused = gauss.ddim_sample(diff, Y_1, Y_2, embeddings=fused_scheme, ddim_timesteps=1,
        #                           desc=f'Generating {names[0]:>20s}')
        fused = gauss.ddpm_sample(diff, Y_1, Y_2, embeddings=fused_scheme, desc=f'Generating {names[0]:>20s}')
        Y_1, Y_2, fused = denorm(Y_1), denorm(Y_2), denorm(fused)

        Qabf.append(metric_Qabf(Y_1, Y_2, fused))
        psnr.append(metric_psnr(Y_1, Y_2, fused))
        ssim.append(metric_ssim(Y_1, Y_2, fused))

        fused = ycbcr2rgb(fused, Cb, Cr)
        fused = torch.clamp(fused, 0, 1)

        fused = fused.squeeze(0).cpu()
        writer.add_image(f'mefb', fused, global_step=i)
        fused = to_pil(fused)
        fused.save(os.path.join(save_path, 'mefb', names[0]))

    writer.add_scalar('psnr/1_mefb', sum(psnr) / len(psnr))
    writer.add_scalar('ssim/1_mefb', sum(ssim) / len(ssim))
    writer.add_scalar('Qabf/1_mefb', sum(Qabf) / len(Qabf))

    print("\nGenerating MSRS ...")

    psnr = []
    ssim = []
    Qabf = []
    fused_scheme = torch.tensor([2]).to(args.device)
    for i, (S1, S2, names) in enumerate(msrs_dataloader):
        S1, S2 = S1.to(args.device), S2.to(args.device)
        Y_1 = S1
        Y_2, Cb, Cr = rgb2ycbcr(S2)
        Y_1, Y_2 = norm(Y_1), norm(Y_2)

        # fused = gauss.ddim_sample(diff, Y_1, Y_2, embeddings=fused_scheme, ddim_timesteps=1,
        #                           desc=f'Generating {names[0]}')
        fused = gauss.ddpm_sample(diff, Y_1, Y_2, embeddings=fused_scheme, desc=f'Generating {names[0]}')
        Y_1, Y_2, fused = denorm(Y_1), denorm(Y_2), denorm(fused)

        Qabf.append(metric_Qabf(Y_1, Y_2, fused))
        psnr.append(metric_psnr(Y_1, Y_2, fused))
        ssim.append(metric_ssim(Y_1, Y_2, fused))

        fused = ycbcr2rgb(fused, Cb, Cr)
        fused = torch.clamp(fused, 0, 1)

        fused = fused.squeeze(0).cpu()
        writer.add_image(f'msrs', fused, global_step=i)
        fused = to_pil(fused)
        fused.save(os.path.join(save_path, 'msrs', names[0]))

    writer.add_scalar('psnr/2_msrs', sum(psnr) / len(psnr))
    writer.add_scalar('ssim/2_msrs', sum(ssim) / len(ssim))
    writer.add_scalar('Qabf/2_msrs', sum(Qabf) / len(Qabf))

    print("\nGenerating CT-MRI ...")

    psnr = []
    ssim = []
    Qabf = []
    fused_scheme = torch.tensor([3]).to(args.device)
    for i, (S1, S2, names) in enumerate(ct_mri_dataloader):
        S1, S2 = S1.to(args.device), S2.to(args.device)
        S1, S2 = norm(S1), norm(S2)
        fused = gauss.ddim_sample(diff, S1, S2, embeddings=fused_scheme, ddim_timesteps=50,
                                  desc=f'Generating {names[0]}')
        fused = gauss.ddpm_sample(diff, S1, S2, embeddings=fused_scheme, desc=f'Generating {names[0]}')
        S1, S2, fused = denorm(S1), denorm(S2), denorm(fused)

        Qabf.append(metric_Qabf(S1, S2, fused))
        psnr.append(metric_psnr(S1, S2, fused))
        ssim.append(metric_ssim(S1, S2, fused))

        fused = torch.clamp(fused, 0, 1)

        fused = fused.squeeze(0).cpu()
        writer.add_image(f'ct_mri', fused, global_step=i)
        fused = to_pil(fused)
        fused.save(os.path.join(save_path, 'ct_mri', names[0]))

    writer.add_scalar('psnr/3_ct-mri', sum(psnr) / len(psnr))
    writer.add_scalar('ssim/3_ct-mri', sum(ssim) / len(ssim))
    writer.add_scalar('Qabf/3_ct-mri', sum(Qabf) / len(Qabf))

    print("\nGenerating PET-MRI ...")

    psnr = []
    ssim = []
    Qabf = []
    fused_scheme = torch.tensor([4]).to(args.device)
    for i, (S1, S2, names) in enumerate(pet_mri_dataloader):
        S1, S2 = S1.to(args.device), S2.to(args.device)
        Y_1, Cb, Cr = rgb2ycbcr(S1)
        Y_2 = S2
        Y_1, Y_2 = norm(Y_1), norm(Y_2)

        # fused = gauss.ddim_sample(diff, Y_1, Y_2, embeddings=fused_scheme, ddim_timesteps=50,
        #                           desc=f'Generating {names[0]}')
        fused = gauss.ddpm_sample(diff, Y_1, Y_2, embeddings=fused_scheme, desc=f'Generating {names[0]}')
        Y_1, Y_2, fused = denorm(Y_1), denorm(Y_2), denorm(fused)

        Qabf.append(metric_Qabf(Y_1, Y_2, fused))
        psnr.append(metric_psnr(Y_1, Y_2, fused))
        ssim.append(metric_ssim(Y_1, Y_2, fused))

        fused = ycbcr2rgb(fused, Cb, Cr)
        fused = torch.clamp(fused, 0, 1)

        fused = fused.squeeze(0).cpu()
        writer.add_image(f'pet_mri', fused, global_step=i)
        fused = to_pil(fused)
        fused.save(os.path.join(save_path, 'pet_mri', names[0]))

    writer.add_scalar('psnr/4_pet-mri', sum(psnr) / len(psnr))
    writer.add_scalar('ssim/4_pet-mri', sum(ssim) / len(ssim))
    writer.add_scalar('Qabf/4_pet-mri', sum(Qabf) / len(Qabf))

    writer.close()
