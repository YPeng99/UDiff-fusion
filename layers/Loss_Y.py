import torch
import numpy as np
import torch.nn as nn
from kornia.losses import ssim_loss
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, weights=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.weights = weights if weights is not None else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.channel = 1

        # 创建高斯窗
        self.window = nn.Parameter(self.create_window(window_size), requires_grad=False)

    def create_window(self, window_size):
        gauss = torch.Tensor([np.exp(-0.5 * (x - (window_size - 1) / 2) ** 2 / (0.5 ** 2)) for x in range(window_size)])
        window = gauss / gauss.sum()
        window = window.view(1, -1) * window.view(-1, 1)
        return window.unsqueeze(0).unsqueeze(0)  # 扩展到 (1, 1, window_size)

    def forward(self, img1, img2):
        # return self.ssim(img1, img2)
        return self.msssim(img1, img2)

    def msssim(self, img1, img2):
        msssim = 1
        for weight in self.weights:
            ssim_value = self.ssim(img1, img2) ** weight
            msssim *= ssim_value
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        return msssim

    def ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C = 0.03 ** 2
        ssim_map = (2 * sigma12 + C) / (sigma1_sq + sigma2_sq + C)
        if self.size_average:
            # return F.l1_loss((sigma1_sq + sigma2_sq), 2 * sigma12, reduction='mean')
            return 1 - ssim_map.mean()
        else:
            # return F.l1_loss((sigma1_sq + sigma2_sq), 2 * sigma12, reduction='none')
            return 1 - ssim_map


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
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self, threshold=0., channels=1):
        super().__init__()
        self.threshold = threshold
        self.channels = channels
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.ssim_loss = SSIMLoss(size_average=True)

    def forward(self, image_A, image_B, image_fused):
        ssim_loss = 0.5 * self.ssim_loss(image_A, image_fused) + 0.5 * self.ssim_loss(image_B, image_fused)
        return ssim_loss


class L_Inten(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1

        # 创建高斯窗
        self.window = nn.Parameter(self.create_window(window_size), requires_grad=False)

    def create_window(self, window_size):
        gauss = torch.Tensor([np.exp(-0.5 * (x - (window_size - 1) / 2) ** 2 / (0.5 ** 2)) for x in range(window_size)])
        window = gauss / gauss.sum()
        window = window.view(1, -1) * window.view(-1, 1)
        return window.unsqueeze(0).unsqueeze(0)  # 扩展到 (1, 1, window_size)

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
        return F.l1_loss(mu1, mu2, reduction='none')


#
# class fusion_loss(nn.Module):
#     """无监督LOSS"""
#
#     def __init__(self):
#         super().__init__()
#         self.L_Inten = L_Inten()
#         # self.L_Inten = nn.L1Loss(reduction='none')
#         # self.sobelconv = Sobelxy()
#         # self.L_Grad = L_Grad()
#         self.L_SSIM = L_SSIM()
#
#     def forward(self, image_A, image_B, image_fused, fuse_scheme):
#         mef_index = fuse_scheme == 0
#         mff_index = fuse_scheme == 1
#         vif_index = fuse_scheme == 2
#
#         s_loss = self.L_SSIM(image_A, image_B, image_fused)
#         mef_loss = self.L_Inten(image_fused[mef_index], (image_A[mef_index] + image_B[mef_index]) / 2)
#         mff_loss = self.L_Inten(image_fused[mff_index], torch.max(image_A[mff_index], image_B[mff_index]))
#         vif_loss = self.L_Inten(image_fused[vif_index], torch.max(image_A[vif_index], image_B[vif_index]))
#         fusion_loss = torch.mean(torch.cat([mef_loss, mff_loss, vif_loss], dim=0)) + s_loss

# l1_loss = self.L_Inten(image_fused[mean_imdex], (image_A[mean_imdex] + image_B[mean_imdex]) / 2) + \
#           self.L_Inten(image_fused[max_imdex], torch.max(image_A[max_imdex], image_B[max_imdex])) + \
#         self.L_Inten

# l1_loss = F.l1_loss((image_A+image_B)/2,image_fused)
# g_loss = self.L_Grad(image_A, image_B, image_fused)
# s_loss = self.L_SSIM(image_A, image_B, image_fused)
# fusion_loss = 1. * l1_loss  + 1. * s_loss
# fusion_loss = 0.5 * ssim_loss(image_A,image_fused,11) + 0.5 *ssim_loss(image_B,image_fused,11)
# return fusion_loss

class fusion_loss(nn.Module):
    """无监督LOSS"""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        self.sobelconv = Sobelxy()

        # 创建高斯窗
        self.window = nn.Parameter(self.create_window(window_size), requires_grad=False)

    def create_window(self, window_size):
        gauss = torch.Tensor([np.exp(-0.5 * (x - (window_size - 1) / 2) ** 2 / (0.5 ** 2)) for x in range(window_size)])
        window = gauss / gauss.sum()
        window = window.view(1, -1) * window.view(-1, 1)
        return window.unsqueeze(0).unsqueeze(0)  # 扩展到 (1, 1, window_size)

    def forward(self, image_A, image_B, image_fused, fuse_scheme):
        mef_index = fuse_scheme == 0
        mff_index = fuse_scheme == 1
        # vif_index = fuse_scheme == 1

        image_AB = torch.zeros_like(image_fused)
        image_AB[mef_index] = (image_A[mef_index] + image_B[mef_index]) / 2
        image_AB[mff_index] = torch.max(image_A[mff_index], image_B[mff_index])
        # image_AB[vif_index] = torch.max(image_A[vif_index], image_B[vif_index])

        l_loss = F.l1_loss(image_AB, image_fused)

        mu_A = F.conv2d(image_A, self.window, padding=self.window_size // 2, groups=self.channel)
        mu_B = F.conv2d(image_B, self.window, padding=self.window_size // 2, groups=self.channel)
        mu_fused = F.conv2d(image_fused, self.window, padding=self.window_size // 2, groups=self.channel)

        mu_A_sq = mu_A ** 2
        mu_B_sq = mu_B ** 2
        mu_fused_sq = mu_fused ** 2

        mu_A_mu_f = mu_A * mu_fused
        mu_B_mu_f = mu_B * mu_fused

        sigma_A_sq = F.conv2d(image_A ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu_A_sq
        sigma_B_sq = F.conv2d(image_B ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu_B_sq
        sigma_fused_sq = F.conv2d(image_fused ** 2, self.window, padding=self.window_size // 2,
                                  groups=self.channel) - mu_fused_sq

        sigma_Af = F.conv2d(image_A * image_fused, self.window, padding=self.window_size // 2,
                            groups=self.channel) - mu_A_mu_f
        sigma_Bf = F.conv2d(image_B * image_fused, self.window, padding=self.window_size // 2,
                            groups=self.channel) - mu_B_mu_f

        C = 0.03 ** 2
        cs1_map = (2 * sigma_Af + C) / (sigma_A_sq + sigma_fused_sq + C)
        cs2_map = (2 * sigma_Bf + C) / (sigma_B_sq + sigma_fused_sq + C)

        cs_map = (0.5 * cs1_map + 0.5 * cs2_map)
        cs_loss = 1 - (cs_map.mean() + 1) / 2

        grad_loss = F.l1_loss(self.sobelconv(image_fused), torch.max(self.sobelconv(image_A), self.sobelconv(image_B)))
        return l_loss + 0. * cs_loss + grad_loss


if __name__ == '__main__':
    S1 = torch.randn((8, 1, 224, 224))
    S2 = torch.randn((8, 1, 224, 224))
    fused = torch.randn((8, 1, 224, 224))
    fuse_scheme = torch.randint(0, 3, (8,))
    loss = fusion_loss()
    S1 = S1.clamp(-1, 1)
    l = loss(S1, S1, S1, fuse_scheme)
    print(l)
