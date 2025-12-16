import torch.nn as nn
from kornia.losses import SSIMLoss

from layers.mef_ssim import MEF_SSIM_Loss
from einops import rearrange, repeat

import torch
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self, channel=1):
        super().__init__()
        kernelx = torch.Tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        kernely = torch.Tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
        kernelx = repeat(kernelx, "h w -> b c h w", b=1, c=channel)
        kernely = repeat(kernely, "h w -> b c h w", b=1, c=channel)

        self.register_buffer('kernelx', kernelx)
        self.register_buffer('kernely', kernely)

    def forward(self, x, isSignGrad=True):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        sobelx = F.conv2d(x, self.kernelx, padding=0)
        sobely = F.conv2d(x, self.kernely, padding=0)
        if isSignGrad:
            return sobelx + sobely
        else:
            return torch.abs(sobelx) + torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self, kernel, reduction='mean'):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, image_a, image_b, image_fused,isSignGrad=False):
        gradient_a = self.kernel(image_a,isSignGrad)
        gradient_b = self.kernel(image_b,isSignGrad)
        gradient_fused = self.kernel(image_fused,isSignGrad)

        mask = torch.ge(torch.abs(gradient_a), torch.abs(gradient_b))
        gradient_joint = gradient_a.masked_fill_(~mask, 0.) + gradient_b.masked_fill_(mask, 0.)

        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint, reduction=self.reduction)
        return Loss_gradient


class FusionLoss(nn.Module):
    """无监督LOSS"""

    def __init__(self, channel=1):
        super().__init__()
        self.L_Inten = nn.L1Loss(reduction='mean')
        self.L_Grad = L_Grad(kernel=Sobelxy(channel), reduction='mean')
        self.L_SSIM = MEF_SSIM_Loss(window_size=11, channel=channel, max_val=2.0)
        self.LM_SSIM = SSIMLoss(window_size=11, max_val=2.0)

        # MFF（聚焦)
        self.w_inten_mff = 0.2
        self.w_grad_mff = 0.4
        self.w_ssim_mff = 0.4

        # MEF（曝光)
        self.w_inten_mef = 0.2
        self.w_grad_mef = 0.4
        self.w_ssim_mef = 0.4

        # VIF（红外可见)
        self.w_inten_vif = 0.2
        self.w_grad_vif = 0.4
        self.w_ssim_vif = 0.4

        # MED（医学）
        self.w_inten_med = 0.2
        self.w_grad_med = 0.4
        self.w_ssim_med = 0.4

    def forward(self, image_a, image_b, image_fused, task_index: torch.LongTensor):
        # is_warmup = False

        # mff loss
        mff_index = task_index == 0
        if mff_index.any().item():
            mff_a = image_a[mff_index]
            mff_b = image_b[mff_index]
            mff_fused = image_fused[mff_index]

            # mff_X = rearrange(mff_fused, 'b c h w -> c (h b) w').unsqueeze(0)
            # mff_Ys = torch.stack([rearrange(img, 'b c h w -> c (h b) w') for img in [mff_a, mff_b]], dim=0)

            mff_inten_loss = self.L_Inten(mff_fused, torch.max(mff_a, mff_b))
            mff_grad_loss = self.L_Grad(mff_a, mff_b, mff_fused)
            # mff_ssim_loss = self.L_SSIM(mff_X, mff_Ys, is_lum=False, cen_lum=0.)
            # mff_ssim_loss = mff_ssim_loss.mean()
            mff_ssim_loss = 0.5 * self.LM_SSIM(mff_a, mff_fused) + 0.5 * self.LM_SSIM(mff_b, mff_fused)

            mff_total_loss = self.w_inten_mff * mff_inten_loss + self.w_grad_mff * mff_grad_loss + self.w_ssim_mff * mff_ssim_loss
        else:
            mff_total_loss = torch.tensor(0.).to(image_fused)

        # mef loss
        mef_index = task_index == 1
        if mef_index.any().item():
            mef_a = image_a[mef_index]
            mef_b = image_b[mef_index]
            mef_fused = image_fused[mef_index]

            mef_X = rearrange(mef_fused, 'b c h w -> c (h b) w').unsqueeze(0)
            mef_Ys = torch.stack([rearrange(img, 'b c h w -> c (h b) w') for img in [mef_a, mef_b]], dim=0)

            mef_inten_loss = self.L_Inten(mef_fused, (mef_a + mef_b) / 2)
            mef_grad_loss = self.L_Grad(mef_a, mef_b, mef_fused)
            mef_ssim_loss = self.L_SSIM(mef_X, mef_Ys, is_lum=True, cen_lum=0.)
            mef_ssim_loss = mef_ssim_loss.mean()
            mef_total_loss = self.w_inten_mef * mef_inten_loss + self.w_grad_mef * mef_grad_loss + self.w_ssim_mef * mef_ssim_loss
        else:
            mef_total_loss = torch.tensor(0.).to(image_fused)

        # vif loss
        vif_index = task_index == 2
        if vif_index.any().item():
            vif_a = image_a[vif_index]
            vif_b = image_b[vif_index]
            vif_fused = image_fused[vif_index]

            vif_X = rearrange(vif_fused, 'b c h w -> c (h b) w').unsqueeze(0)
            vif_Ys = torch.stack([rearrange(img, 'b c h w -> c (h b) w') for img in [vif_a, vif_b]], dim=0)

            vif_inten_loss = self.L_Inten(vif_fused, torch.max(vif_a, vif_b))
            vif_grad_loss = self.L_Grad(vif_a, vif_b, vif_fused)
            vif_ssim_loss = self.L_SSIM(vif_X, vif_Ys, is_lum=True, cen_lum=0.)

            vif_total_loss = self.w_inten_vif * vif_inten_loss + self.w_grad_vif * vif_grad_loss + self.w_ssim_vif * vif_ssim_loss
        else:
            vif_total_loss = torch.tensor(0.).to(image_fused)

        # med loss
        med_index = (task_index == 3) | (task_index == 4)
        if med_index.any().item():
            med_a = image_a[med_index]
            med_b = image_b[med_index]
            med_fused = image_fused[med_index]

            med_X = rearrange(med_fused, 'b c h w -> c (h b) w').unsqueeze(0)
            med_Ys = torch.stack([rearrange(img, 'b c h w -> c (h b) w') for img in [med_a, med_b]], dim=0)

            med_inten_loss = self.L_Inten(med_fused, torch.max(med_a, med_b))
            med_grad_loss = self.L_Grad(med_a, med_b, med_fused)
            mef_ssim_loss = self.L_SSIM(med_X, med_Ys, is_lum=False, cen_lum=0.)
            med_total_loss = self.w_inten_med * med_inten_loss + self.w_grad_med * med_grad_loss + self.w_ssim_med * mef_ssim_loss

        else:
            med_total_loss = torch.tensor(0.).to(image_fused)

        loss_list = [mff_total_loss, mef_total_loss, vif_total_loss, med_total_loss]
        non_zero_losses = [l for l in loss_list if l.item() != 0.]
        return sum(non_zero_losses) / len(non_zero_losses)


if __name__ == '__main__':
    S1 = torch.rand([8, 3, 224, 224])
    S2 = torch.rand([8, 3, 224, 224])
    fused = torch.rand([8, 3, 224, 224])
    S1 = S1 * 2 - 1
    S2 = S2 * 2 - 1
    fused = fused * 2 - 1
    task_type = torch.randint(0, 5, [8, ])
    S1 = S1.cuda()
    S2 = S2.cuda()
    fused = fused.cuda()
    task_type = task_type.cuda()

    loss = FusionLoss(channel=3)
    loss.cuda()
    l = loss(S1, S2, fused, task_type)
    print(l)
