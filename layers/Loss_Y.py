import torch
import numpy as np
import torch.nn as nn
from kornia.losses import ssim_loss
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        sobelx = F.conv2d(x, self.weightx, padding=0)
        sobely = F.conv2d(x, self.weighty, padding=0)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self,reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint,reduction=self.reduction)
        return Loss_gradient


class FusionLoss(nn.Module):
    """无监督LOSS"""

    def __init__(self):
        super().__init__()
        self.L_Inten = nn.L1Loss(reduction='none')
        self.L_Grad = L_Grad(reduction='mean')

    def forward(self, image_A, image_B, image_fused, fuse_scheme):
        mean_index = fuse_scheme == 1
        max_index = ~mean_index

        mean_loss = self.L_Inten(image_fused[mean_index], (image_A[mean_index] + image_B[mean_index]) / 2)
        max_loss = self.L_Inten(image_fused[max_index], torch.max(image_A[max_index], image_B[max_index]))
        int_loss = torch.cat([mean_loss,max_loss],dim=0).mean()
        g_loss = self.L_Grad(image_A, image_B, image_fused)
        s_loss = 0.5 * ssim_loss(image_A, image_fused,window_size=11) + 0.5 *ssim_loss(image_B, image_fused,window_size=11)
        fusion_loss = 1. * int_loss + 2. * g_loss + 0. * s_loss

        return fusion_loss


if __name__ == '__main__':
    S1 = torch.randn((8, 1, 224, 224))
    S2 = torch.randn((8, 1, 224, 224))

    noise = torch.randn((8, 1, 224, 224))
    fused = torch.randn((8, 1, 224, 224))
    fuse_scheme = torch.randint(0, 3, (8,))
    loss = FusionLoss()
    l = loss(S1, S2, fused, fuse_scheme)
    print(l)
